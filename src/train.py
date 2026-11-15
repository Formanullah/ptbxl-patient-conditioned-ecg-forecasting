<<<<<<< HEAD
"""Validation-selected, resumable training. Test outputs are never tuning inputs."""

import argparse
import copy
=======
"""One training loop, validation checkpoint selection, and exact-state resume."""

import copy
import os
>>>>>>> origin/work/manual-src-workflow
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
<<<<<<< HEAD
from filelock import FileLock
from torch.nn import functional as F

from config import (
    VARIANTS,
    digest,
    file_hash,
    load_config,
    read_json,
    run_directory,
    runtime,
    write_json,
)
from evaluate import learning_curves
from models import augment, build_model, nt_xent
from preprocessing import make_loaders, validate_cache


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
=======
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
>>>>>>> origin/work/manual-src-workflow
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


<<<<<<< HEAD
def atomic_checkpoint(path, state):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


=======
>>>>>>> origin/work/manual-src-workflow
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


<<<<<<< HEAD
def epoch(model, loader, device, cfg, optimizer=None, contrastive=False):
    training = optimizer is not None
    model.train(training)
    total_mse, total_aux, count, valid, skipped = 0.0, 0.0, 0, 0, 0
    physical = "history_seconds" in cfg["data"]
    records = []
    if physical:
        manifest = read_json(Path(cfg["data"]["cache"]) / "manifest.json")
        shape = (1, 1, 12) if cfg["data"].get("layout") == "time_first" else (1, 12, 1)
        scale = torch.tensor(manifest["std_mV"], device=device).reshape(shape)
    for batch in loader:
        x, y, static = [batch[k].to(device) for k in ("x", "y", "static")]
        with torch.set_grad_enabled(training):
            pred = model(x, static)
            mse = F.mse_loss(pred, y)
            aux = mse.new_zeros(())
            anchors = 0
            if training and contrastive:
                settings = cfg["contrastive"]
                a, b = model.project(augment(x, settings)), model.project(augment(x, settings))
                aux, anchors = nt_xent(a, b, batch["patient_id"], settings["temperature"])
            loss = forecast_loss(pred, y, cfg) + cfg["contrastive"]["weight"] * aux
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss; training stopped")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                optimizer.step()
        n = len(x)
        total_mse += mse.item() * n
        total_aux += aux.item() * anchors
        count += n
        valid += anchors
        skipped += (2 * n - anchors) if training and contrastive else 0
        if physical:
            error = (pred.detach() - y) * scale
            mae = error.abs().mean(dim=(1, 2)).cpu().numpy()
            pmse = error.square().mean(dim=(1, 2)).cpu().numpy()
            records.extend(
                zip(batch["patient_id"].tolist(), mae.tolist(), pmse.tolist(), strict=True)
            )
    if not count:
        raise ValueError("Empty DataLoader")
    result = {
        "mse": total_mse / count,
        "contrastive_loss": total_aux / max(valid, 1),
        "valid_anchors": valid,
        "skipped_anchors": skipped,
    }
    if physical:
        frame = pd.DataFrame(records, columns=["patient", "mae", "mse"])
        result.update(
            mae=float(frame.mae.mean()),
            mse_mV2=float(frame.mse.mean()),
            rmse=float(np.sqrt(frame.mse.mean())),
            patient_macro_mae=float(frame.groupby("patient").mae.mean().mean()),
        )
    return result


def forecast_loss(pred, target, cfg):
    settings = cfg["training"]
    kind = settings.get("loss", "mse")
    if kind == "mse":
        loss = F.mse_loss(pred, target)
    elif kind == "huber":
        loss = F.huber_loss(pred, target, delta=settings.get("huber_delta", 1.0))
    else:
        raise ValueError("Loss must be mse or huber")
    axis = 1 if cfg["data"].get("layout") == "time_first" else -1
    if settings.get("derivative_weight", 0):
        loss = loss + settings["derivative_weight"] * F.l1_loss(
            torch.diff(pred, dim=axis), torch.diff(target, dim=axis)
        )
    if settings.get("spectral_weight", 0):
        loss = loss + settings["spectral_weight"] * F.l1_loss(
            torch.fft.rfft(pred, dim=axis).abs(), torch.fft.rfft(target, dim=axis).abs()
        )
    return loss


def train(cfg, architecture, variant, resume=False):
    manifest = validate_cache(cfg)
    cfg = copy.deepcopy(cfg)
    if (
        cfg.get("experiment", {}).get("architecture", architecture) != architecture
        or cfg.get("experiment", {}).get("variant", variant) != variant
    ):
        raise ValueError("Model/variant differ from the named experiment design")
    seed_everything(cfg["training"]["seed"])
    device = runtime(cfg)
    run_name = f"{architecture}_{variant}"
    run = run_directory(cfg, architecture, variant)
    run.mkdir(parents=True, exist_ok=True)
    scientific_config = copy.deepcopy(cfg)
    scientific_config.pop("paths")
    scientific_config.pop("storage", None)
    for field in ("root", "cache"):
        scientific_config["data"].pop(field)
    for field in ("device", "threads", "num_workers"):
        scientific_config["training"].pop(field)
    signature_data = {
        "config": scientific_config,
        "architecture": architecture,
        "variant": variant,
        "fingerprint": manifest["fingerprint"],
        "smoke": False,
    }
    if "history_seconds" in cfg["data"]:
        signature_data["feature_selection_hash"] = {
            key: file_hash(value)
            for key, value in cfg.get("features", {}).items()
            if key == variant + "_path" and variant in ("mrmr", "permutation")
        }
    signature = digest(signature_data)
    if (run / "config.json").exists():
        previous = read_json(run / "config.json")
        if previous["signature"] != signature:
            raise ValueError(f"Configuration changed for {run}; use a new experiments directory")
    if (run / "last.pt").exists() and not resume:
        raise ValueError(
            f"Saved run exists at {run}. Use --resume to continue it, "
            "or choose new experiment paths for fresh training."
        )
    if resume and not (run / "last.pt").exists():
        raise ValueError(f"No checkpoint to resume at {run}; omit --resume for a new run")
    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    loaders = make_loaders(cfg, variant, generator, splits=("train", "val"))
    static_dim = loaders["train"].dataset.static_dim
    model = build_model(cfg, architecture, static_dim).to(device)
    params = sum(p.numel() for p in model.parameters())
    trained_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if not name.startswith("projector.") or variant.endswith("contrastive")
    )
    config = {
        "config": cfg,
        "architecture": architecture,
        "variant": variant,
        "smoke": False,
        "signature": signature,
        "fingerprint": manifest["fingerprint"],
        "selected_features": loaders["train"].dataset.selected_features,
        "parameters": params,
        "parameters_used_by_loss": trained_params,
        "device": str(device),
        "processor": platform.processor(),
        "source_sha256": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
        "tensor_dimensions": {
            "input": [None, cfg["data"]["history"], 12]
            if cfg["data"].get("layout") == "time_first"
            else [None, 12, cfg["data"]["history"]],
            "target": [None, cfg["data"]["horizon"], 12]
            if cfg["data"].get("layout") == "time_first"
            else [None, 12, cfg["data"]["horizon"]],
            "static": [None, static_dim],
            "dynamic_state": [None, cfg["model"]["hidden"]],
        },
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ["torch", "numpy", "pandas", "wfdb", "matplotlib"]},
    }
    write_json(run / "config.json", config)
    write_json(
        run / "feature_manifest.json",
        {"features": loaders["train"].dataset.selected_features, "preprocessing": manifest},
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        if cfg["training"].get("scheduler") == "plateau"
        else None
    )
    research = "history_seconds" in cfg["data"]
    best, bad_epochs, history, elapsed = float("inf"), 0, [], 0.0
    if resume:
        # Only load locally generated, trusted checkpoints: includes Python/NumPy RNG state.
        state = torch.load(run / "last.pt", map_location=device, weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Checkpoint configuration mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        best, bad_epochs, history, elapsed = [
            state[k] for k in ("best", "bad_epochs", "history", "elapsed")
        ]
        restore_rng(state["rng"], generator)
        print(f"RESUME {run_name} after epoch {len(history)}", flush=True)
    started = time.perf_counter()
    max_epochs = cfg["training"]["epochs"]
    for number in range(len(history) + 1, max_epochs + 1):
        if (
            cfg["training"].get("early_stopping", True)
            and bad_epochs >= cfg["training"]["patience"]
        ):
            break
        tick = time.perf_counter()
        cpu_tick = time.process_time()
        tr = epoch(
            model,
            loaders["train"],
            device,
            cfg,
            optimizer,
            contrastive=variant.endswith("contrastive"),
        )
        va = epoch(model, loaders["val"], device, cfg)
        row = {
            "epoch": number,
            "train_mse": tr["mse"],
            "val_mse": va["mse"],
            "contrastive_loss": tr["contrastive_loss"],
            "valid_anchors": tr["valid_anchors"],
            "skipped_anchors": tr["skipped_anchors"],
            "seconds": time.perf_counter() - tick,
            "cpu_seconds": time.process_time() - cpu_tick,
        }
        if research:
            row.update(
                {
                    f"{prefix}_{key}": metrics[key]
                    for prefix, metrics in (("train", tr), ("val", va))
                    for key in ("mae", "rmse", "mse_mV2", "patient_macro_mae")
                }
            )
            row["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        objective = va["patient_macro_mae"] if research else va["mse"]
        if scheduler is not None:
            scheduler.step(objective)
        if objective < best:
            best, bad_epochs = objective, 0
            atomic_checkpoint(
                run / "best.pt",
                {
                    "model": model.state_dict(),
                    "epoch": number,
                    "val_mse": va["mse"],
                    "selection_metric": "patient_macro_mae_mV" if research else "mse_normalized",
                    "selection_value": best,
                    "signature": signature,
                },
            )
        else:
            bad_epochs += 1
        atomic_checkpoint(
            run / "last.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best": best,
                "bad_epochs": bad_epochs,
                "history": history,
                "elapsed": elapsed + time.perf_counter() - started,
                "rng": rng_state(generator),
                "signature": signature,
            },
        )
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(f"{run_name} {row}", flush=True)
    elapsed += time.perf_counter() - started
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    summary = {
        "best_epoch": checkpoint["epoch"],
        "best_val_mse": checkpoint["val_mse"],
        "epochs_completed": len(history),
        "training_seconds": elapsed,
        "parameters": params,
        "parameters_used_by_loss": trained_params,
        "architecture": architecture,
        "variant": variant,
        "smoke": False,
        "fingerprint": manifest["fingerprint"],
    }
    write_json(run / "training_metrics.json", summary)
    learning_curves(run, cfg)
    if research:
        summary.update(
            selection_metric="validation_patient_macro_mae_mV",
            selection_value=checkpoint["selection_value"],
            final_validation={k: v for k, v in history[-1].items() if k.startswith("val_")},
        )
        write_json(run / "training_metrics.json", summary)
        print(
            f"TRAINED {run.name}: selected validation patient-macro MAE={checkpoint['selection_value']:.6f} mV. Test remains reserved.",
            flush=True,
        )
    else:
        print(
            f"TRAINED {run_name}: best validation MSE={summary['best_val_mse']:.6f}. Run evaluate.py for test metrics.",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--model", "-model", choices=["lstm", "cnn_lstm"])
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--resume", action="store_true", help="Continue the chosen saved checkpoint"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.experiment)
    if args.device:
        config["training"]["device"] = args.device

    with FileLock(".pipeline.lock", timeout=0):
        train(
            config,
            args.model or config.get("experiment", {}).get("architecture", "lstm"),
            args.variant or config.get("experiment", {}).get("variant", "patient_history"),
            resume=args.resume,
        )
=======
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
>>>>>>> origin/work/manual-src-workflow
