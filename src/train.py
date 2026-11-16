"""One training loop, validation checkpoint selection, and exact-state resume."""

import copy
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from config import (
    SCHEMA,
    device_for,
    digest,
    file_hash,
    read_json,
    source_hashes,
    study_dir,
    write_json,
)
from data import ECGDataset, loader
from models import build_model
from preprocessing import feature_arrays, prepare_transforms


def seed_everything(seed, threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def static_inputs(cfg, cache, feature_set, selection=None, include_test=False):
    base = "expanded" if feature_set in ("mrmr", "permutation") else feature_set
    arrays, names, details = feature_arrays(
        cfg, cache, base, splits=("train", "val", "test") if include_test else ("train", "val")
    )
    if selection is not None:
        if selection["cohort"] != read_json(Path(cache) / "manifest.json")["cohort"]:
            raise ValueError("Feature selection belongs to another cohort")
        indices = [names.index(name) for name in selection["selected_features"]]
        arrays = {split: a[:, indices] for split, a in arrays.items()}
        names = selection["selected_features"]
        details["selection"] = selection
    elif feature_set in ("mrmr", "permutation"):
        raise ValueError("Selected-feature runs require a provenance-checked selection")
    return arrays, names, details


def epoch(model, batches, device, std, optimizer=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    records, total, count = [], 0.0, 0
    scale = torch.as_tensor(std, dtype=torch.float32, device=device)
    for batch in batches:
        x, y, static = [batch[k].to(device) for k in ("x", "y", "static")]
        with torch.set_grad_enabled(training):
            prediction = model(x, static)
            loss = F.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        error = (prediction.detach() - y) * scale
        records.extend(
            zip(
                batch["patient_id"].tolist(),
                error.abs().mean((1, 2)).cpu().tolist(),
                error.square().mean((1, 2)).cpu().tolist(),
                strict=True,
            )
        )
        total += loss.item() * len(x)
        count += len(x)
    if not count:
        raise ValueError("Empty DataLoader")
    frame = pd.DataFrame(records, columns=["patient", "mae", "mse"])
    return {
        "mse_normalized": total / count,
        "mae_mV": float(frame.mae.mean()),
        "rmse_mV": float(np.sqrt(frame.mse.mean())),
        "patient_macro_mae_mV": float(frame.groupby("patient").mae.mean().mean()),
    }


def train_run(
    cfg,
    cache,
    architecture,
    preprocessing="P0",
    feature_set="F0",
    *,
    name=None,
    epochs=None,
    lr=None,
    hidden=None,
    resume=False,
    selection=None,
    stop_after=None,
):
    cfg = copy.deepcopy(cfg)
    if lr is not None:
        cfg["training"]["lr"] = lr
    if hidden is not None:
        cfg["model"]["hidden"] = hidden
    mode, tr = cfg["run_mode"], cfg["training"]
    budget = epochs or (
        tr["pilot_epochs"]
        if mode == "pilot"
        else (tr["full_epochs"] if mode == "full" else tr["epochs"])
    )
    scaler = prepare_transforms(cache, cfg, preprocessing)
    arrays, names, feature_info = static_inputs(cfg, cache, feature_set, selection)
    manifest = read_json(Path(cache) / "manifest.json")
    seed_everything(tr["seed"], tr["threads"])
    device = device_for(cfg)
    model = build_model(architecture, len(names), cfg["data"]["horizon"], **cfg["model"]).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    name = (
        name
        or f"{architecture}_{preprocessing}_{feature_set}_lr{tr['lr']:g}_h{cfg['model']['hidden']}_e{budget}"
    )
    run = study_dir(cfg) / mode / "checkpoints" / name
    run.mkdir(parents=True, exist_ok=True)
    scientific = {
        "schema": SCHEMA,
        "cohort": manifest["cohort"],
        "preprocessing": preprocessing,
        "filter": cfg["preprocessing"] if preprocessing == "P1" else None,
        "scaler_sha256": file_hash(Path(cache) / "scaler.json"),
        "architecture": architecture,
        "model": cfg["model"],
        "features": names,
        "feature_statistics": feature_info,
        "budget": budget,
        "training": {
            k: tr[k] for k in ("seed", "lr", "batch_size", "weight_decay", "gradient_clip")
        },
        "source": source_hashes(),
        "versions": {p: version(p) for p in ("torch", "numpy", "scipy", "pandas", "wfdb")},
        "synthetic": manifest["synthetic"],
    }
    signature = digest(scientific)
    recorded = (run / "run.json").exists()
    if recorded:
        previous = read_json(run / "run.json")
        if previous.get("schema") != SCHEMA:
            raise ValueError("Legacy checkpoint: use the archived workflow")
        if previous["signature"] != signature:
            raise ValueError("Run configuration/source changed; create a distinct experiment ID")
    if (run / "last.pt").exists() and not resume:
        raise ValueError("Existing run; choose resume explicitly or a new experiment ID")
    if resume and not (run / "last.pt").exists():
        raise ValueError("No last checkpoint to resume")
    generator = torch.Generator().manual_seed(tr["seed"])
    batches = {
        s: loader(ECGDataset(cache, cfg, s, preprocessing, arrays[s]), cfg, s == "train", generator)
        for s in ("train", "val")
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    history, best, elapsed = [], (float("inf"), float("inf")), 0.0
    if resume:
        state = torch.load(run / "last.pt", map_location=device, weights_only=False)
        if state.get("schema") != SCHEMA or state.get("signature") != signature:
            raise ValueError(
                "Checkpoint protocol is incompatible; use the archived workflow for legacy runs"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best, elapsed = state["history"], tuple(state["best"]), state["elapsed"]
        restore_rng(state["rng"], generator)
    record = {
        "schema": SCHEMA,
        "name": name,
        "signature": signature,
        "scientific": scientific,
        "config": cfg,
        "cache": str(Path(cache).resolve()),
        "architecture": architecture,
        "preprocessing": preprocessing,
        "feature_set": feature_set,
        "selected_features": names,
        "selection": selection,
        "parameters": parameters,
        "epoch_budget": budget,
        "smoke": mode == "pilot" or manifest["synthetic"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        },
    }
    if not recorded:
        write_json(run / "run.json", record)
        write_json(run / "cohort_manifest.json", manifest)
        write_json(run / "scaler.json", scaler)
    source_root = Path(__file__).parent
    snapshot = study_dir(cfg) / "source" / digest(scientific["source"])[:16]
    for relative, expected in scientific["source"].items():
        target = snapshot / relative
        if target.exists() and file_hash(target) != expected:
            raise ValueError("Preserved source snapshot changed")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source_root / relative).read_bytes())

    def persist_progress(state):
        number = state["epoch"]
        pd.DataFrame(state["history"]).to_csv(run / "history.csv", index=False)
        write_json(
            run / "status.json",
            {
                "state": "completed" if number == budget else "training incomplete",
                "epochs_completed": number,
                "budget": budget,
                "best_validation_mae_mV": state["best"][0],
                "runtime_seconds": state["elapsed"],
            },
        )

    if resume:
        # last.pt is the epoch transaction; repair derived files only if a write was interrupted.
        status = read_json(run / "status.json") if (run / "status.json").exists() else {}
        if status.get("epochs_completed") != len(history):
            save_checkpoint(run / "best.pt", state["best_checkpoint"])
            if len(history) in (60, 120):
                milestone = {
                    k: state[k] for k in ("schema", "signature", "model", "epoch", "objective")
                }
                save_checkpoint(run / f"epoch_{len(history)}.pt", milestone)
                save_checkpoint(run / f"best_through_{len(history)}.pt", state["best_checkpoint"])
            persist_progress(state)
        best_checkpoint = state["best_checkpoint"]
    else:
        best_checkpoint = None
    started = time.perf_counter()
    for number in range(len(history) + 1, budget + 1):
        tick = time.perf_counter()
        training = epoch(
            model, batches["train"], device, scaler["std"], optimizer, tr["gradient_clip"]
        )
        validation = epoch(model, batches["val"], device, scaler["std"])
        row = {
            "epoch": number,
            "seconds": time.perf_counter() - tick,
            "lr": tr["lr"],
            **{"train_" + k: v for k, v in training.items()},
            **{"val_" + k: v for k, v in validation.items()},
        }
        history.append(row)
        objective = (validation["patient_macro_mae_mV"], validation["rmse_mV"])
        selected = {
            "schema": SCHEMA,
            "signature": signature,
            "model": model.state_dict(),
            "epoch": number,
            "objective": objective,
        }
        if objective < best:
            best = objective
            best_checkpoint = {
                **selected,
                "model": {k: v.detach().cpu().clone() for k, v in selected["model"].items()},
            }
        state = {
            **selected,
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "history": history,
            "best": best,
            "best_checkpoint": best_checkpoint,
            "rng": rng_state(generator),
            "elapsed": elapsed + time.perf_counter() - started,
        }
        save_checkpoint(run / "last.pt", state)
        if best_checkpoint["epoch"] == number:
            save_checkpoint(run / "best.pt", best_checkpoint)
        if number in (60, 120):
            save_checkpoint(run / f"epoch_{number}.pt", selected)
            save_checkpoint(run / f"best_through_{number}.pt", best_checkpoint)
        persist_progress(state)
        print(
            f"{name}: epoch {number}/{budget}; validation patient MAE {objective[0]:.6f} mV",
            flush=True,
        )
        if stop_after is not None and number >= stop_after:
            break
    return run
