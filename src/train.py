"""Validation-selected, resumable training. Test outputs are never tuning inputs."""

import argparse
import copy
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def atomic_checkpoint(path, state):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


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
