"""Validation-selected, resumable training. Test outputs are never tuning inputs."""

import copy
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .config import FEATURES, digest, read_json, write_json
from .data import ECGDataset, make_loaders, validate_cache
from .evaluation import evaluate_arrays, learning_curves
from .models import augment, build_model, nt_xent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def runtime(cfg):
    torch.set_num_threads(cfg["training"]["threads"])
    requested = cfg["training"]["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    return torch.device(device)


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
            loss = mse + cfg["contrastive"]["weight"] * aux
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
    if not count:
        raise ValueError("Empty DataLoader")
    return {
        "mse": total_mse / count,
        "contrastive_loss": total_aux / max(valid, 1),
        "valid_anchors": valid,
        "skipped_anchors": skipped,
    }


def predict(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            predictions.append(
                model(batch["x"].to(device), batch["static"].to(device)).cpu().numpy()
            )
            targets.append(batch["y"].numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def train(cfg, architecture, variant, smoke=False):
    manifest = validate_cache(cfg)
    cfg = copy.deepcopy(cfg)
    if smoke:
        cfg["paths"]["outputs"] = str(Path(cfg["paths"]["outputs"]) / "smoke")
    seed_everything(cfg["training"]["seed"])
    device = runtime(cfg)
    run_name = f"{architecture}_{variant}"
    run = Path(cfg["paths"]["experiments"]) / ("smoke" if smoke else "") / run_name
    run.mkdir(parents=True, exist_ok=True)
    scientific_config = copy.deepcopy(cfg)
    scientific_config.pop("paths")
    for field in ("root", "cache"):
        scientific_config["data"].pop(field)
    for field in ("device", "threads", "num_workers"):
        scientific_config["training"].pop(field)
    signature = digest(
        {
            "config": scientific_config,
            "architecture": architecture,
            "variant": variant,
            "fingerprint": manifest["fingerprint"],
            "smoke": smoke,
        }
    )
    if (run / "config.json").exists():
        previous = read_json(run / "config.json")
        if previous["signature"] != signature:
            raise ValueError(f"Configuration changed for {run}; use a new experiments directory")
        if (run / "test_metrics.json").exists():
            print(f"SKIP completed {run_name}", flush=True)
            return read_json(run / "test_metrics.json")
    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    loaders = make_loaders(cfg, variant, generator, limit=128 if smoke else None)
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
        "smoke": smoke,
        "signature": signature,
        "fingerprint": manifest["fingerprint"],
        "selected_features": FEATURES[:static_dim],
        "parameters": params,
        "parameters_used_by_loss": trained_params,
        "device": str(device),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ["torch", "numpy", "pandas", "wfdb", "matplotlib"]},
    }
    write_json(run / "config.json", config)
    write_json(
        run / "feature_manifest.json",
        {"features": FEATURES[:static_dim], "preprocessing": manifest},
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    best, bad_epochs, history, elapsed = float("inf"), 0, [], 0.0
    if (run / "last.pt").exists():
        # Only load locally generated, trusted checkpoints: includes Python/NumPy RNG state.
        state = torch.load(run / "last.pt", map_location=device, weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Checkpoint configuration mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, bad_epochs, history, elapsed = [
            state[k] for k in ("best", "bad_epochs", "history", "elapsed")
        ]
        restore_rng(state["rng"], generator)
        print(f"RESUME {run_name} after epoch {len(history)}", flush=True)
    started = time.perf_counter()
    max_epochs = min(2, cfg["training"]["epochs"]) if smoke else cfg["training"]["epochs"]
    for number in range(len(history) + 1, max_epochs + 1):
        if bad_epochs >= cfg["training"]["patience"]:
            break
        tick = time.perf_counter()
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
        }
        history.append(row)
        if va["mse"] < best:
            best, bad_epochs = va["mse"], 0
            atomic_checkpoint(
                run / "best.pt",
                {
                    "model": model.state_dict(),
                    "epoch": number,
                    "val_mse": best,
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
    model.load_state_dict(checkpoint["model"])
    pred, true = predict(model, loaders["test"], device)
    mean, std = [
        np.asarray(manifest[k], dtype=np.float32)[None, :, None] for k in ("mean_mV", "std_mV")
    ]
    x = np.asarray(loaders["test"].dataset.x[: len(pred)]) * std + mean
    meta = loaders["test"].dataset.metadata.iloc[: len(pred)].reset_index(drop=True)
    summary = evaluate_arrays(pred * std + mean, true * std + mean, x, meta, run, cfg, manifest)
    summary.update(
        {
            "best_epoch": checkpoint["epoch"],
            "best_val_mse": checkpoint["val_mse"],
            "epochs_completed": len(history),
            "training_seconds": elapsed,
            "parameters": params,
            "parameters_used_by_loss": trained_params,
            "architecture": architecture,
            "variant": variant,
            "smoke": smoke,
        }
    )
    write_json(run / "test_metrics.json", summary)
    learning_curves(run, cfg)
    print(
        f"COMPLETE {run_name}: MAE={summary['mae']:.6f} mV, RMSE={summary['rmse']:.6f} mV",
        flush=True,
    )
    return summary


def baselines(cfg):
    manifest = validate_cache(cfg)
    ds = ECGDataset(cfg, "test", "ecg_only")
    mean, std = [
        np.asarray(manifest[k], dtype=np.float32)[None, :, None] for k in ("mean_mV", "std_mV")
    ]
    x, true = np.asarray(ds.x) * std + mean, np.asarray(ds.y) * std + mean
    horizon = cfg["data"]["horizon"]
    predictions = {
        "persistence": np.repeat(x[:, :, -1:], horizon, axis=2),
        "repeat_window": np.tile(
            x[:, :, -min(horizon, x.shape[-1]) :],
            (1, 1, int(np.ceil(horizon / min(horizon, x.shape[-1])))),
        )[:, :, :horizon],
    }
    for name, pred in predictions.items():
        run = Path(cfg["paths"]["experiments"]) / name
        run.mkdir(parents=True, exist_ok=True)
        if (run / "test_metrics.json").exists():
            if read_json(run / "test_metrics.json")["fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Baseline cache mismatch; use new output directory")
            continue
        write_json(run / "config.json", {"config": cfg, "smoke": False, "baseline": name})
        result = evaluate_arrays(pred, true, x, ds.metadata, run, cfg, manifest)
        print(f"BASELINE {name}: MAE={result['mae']:.6f} mV", flush=True)


def benchmark(cfg):
    """Short real-data timing check, without test-fold evaluation or persisted weights."""
    validate_cache(cfg)
    device = runtime(cfg)
    rows = []
    for architecture in ("lstm", "cnn_lstm"):
        for contrastive in (False, True):
            seed_everything(cfg["training"]["seed"])
            loaders = make_loaders(cfg, "patient_history", limit=256)
            model = build_model(cfg, architecture, 10).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
            started = time.perf_counter()
            epoch(model, loaders["train"], device, cfg, opt, contrastive)
            seconds = time.perf_counter() - started
            full_n = len(ECGDataset(cfg, "train"))
            rows.append(
                {
                    "architecture": architecture,
                    "contrastive": contrastive,
                    "records": 256,
                    "seconds": seconds,
                    "estimated_train_epoch_seconds": seconds * full_n / 256,
                }
            )
    write_json(Path(cfg["paths"]["experiments"]) / "benchmark.json", rows)
    print(rows, flush=True)
    return rows


def evaluate_saved(cfg, architecture, variant):
    """Evaluate an existing selected checkpoint without training or changing weights."""
    manifest = validate_cache(cfg)
    run = Path(cfg["paths"]["experiments"]) / f"{architecture}_{variant}"
    if not (run / "best.pt").exists() or not (run / "config.json").exists():
        raise ValueError(f"No selected checkpoint at {run}; run train first")
    recorded = read_json(run / "config.json")
    if recorded["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Evaluation cache differs from training cache")
    if (run / "test_metrics.json").exists():
        summary = read_json(run / "test_metrics.json")
        print(summary)
        return summary
    device = runtime(cfg)
    loaders = make_loaders(cfg, variant)
    model = build_model(recorded["config"], architecture, loaders["train"].dataset.static_dim).to(
        device
    )
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    pred, true = predict(model, loaders["test"], device)
    mean, std = [
        np.asarray(manifest[k], dtype=np.float32)[None, :, None] for k in ("mean_mV", "std_mV")
    ]
    ds = loaders["test"].dataset
    summary = evaluate_arrays(
        pred * std + mean,
        true * std + mean,
        np.asarray(ds.x) * std + mean,
        ds.metadata,
        run,
        cfg,
        manifest,
    )
    history = pd.read_csv(run / "history.csv")
    summary.update(
        {
            "best_epoch": checkpoint["epoch"],
            "best_val_mse": checkpoint["val_mse"],
            "epochs_completed": len(history),
            "training_seconds": float(history.seconds.sum()),
            "parameters": recorded["parameters"],
            "architecture": architecture,
            "variant": variant,
            "smoke": False,
        }
    )
    write_json(run / "test_metrics.json", summary)
    learning_curves(run, cfg)
    return summary
