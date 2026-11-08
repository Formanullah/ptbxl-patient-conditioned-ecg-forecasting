"""Explicit validation/test inference, peak scoring and waveform figures."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from config import (
    LEADS,
    VARIANTS,
    file_hash,
    load_config,
    read_json,
    run_directory,
    runtime,
    write_json,
)
from metrics import metric_summary, physical_arrays, subgroup_metrics
from models import build_model
from morphology import qrs_metrics
from preprocessing import ECGDataset, make_loaders, validate_cache


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_examples(x, pred, true, meta, out, cfg):
    fs = cfg["data"]["sampling_rate"]
    errors = np.abs(pred - true).mean(axis=(1, 2))
    order = np.argsort(errors, kind="stable")
    selected = {
        "lowest_id": int(np.argmin(meta.ecg_id)),
        "median_error": int(order[len(order) // 2]),
        "worst_error": int(order[-1]),
    }
    for selection, idx in selected.items():
        fig, axes = plt.subplots(6, 2, figsize=(14, 13), sharex=True)
        tx = (np.arange(x.shape[-1]) - x.shape[-1]) / fs
        ty = np.arange(true.shape[-1]) / fs
        for k, ax in enumerate(axes.flat):
            ax.plot(tx, x[idx, k], color="0.6", lw=0.9, label="Observed")
            ax.plot(ty, true[idx, k], lw=1.2, label="Actual")
            ax.plot(ty, pred[idx, k], lw=1.1, label="Forecast")
            ax.axvline(0, color="black", ls="--", lw=0.7)
            ax.set(ylabel=f"{LEADS[k]} (mV)")
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Seconds relative to forecast origin")
        fig.suptitle(f"{selection}: ECG {int(meta.iloc[idx].ecg_id)}, MAE {errors[idx]:.4f} mV")
        save_figure(fig, out / f"forecast_{selection}.png")
    return {name: int(meta.iloc[i].ecg_id) for name, i in selected.items()}


def learning_curves(run, cfg):
    run = Path(run)
    history = pd.read_csv(run / "history.csv")
    out = Path(cfg["paths"]["outputs"]) / run.name
    out.mkdir(parents=True, exist_ok=True)
    contrastive = bool(history.contrastive_loss.abs().sum())
    fig, axes = plt.subplots(
        1, 2 if contrastive else 1, figsize=(12 if contrastive else 7, 4), squeeze=False
    )
    ax = axes[0, 0]
    key = "patient_macro_mae" if "val_patient_macro_mae" in history else "mse"
    ax.plot(history.epoch, history["train_" + key], label="Training")
    ax.plot(history.epoch, history["val_" + key], label="Validation")
    ax.set(
        ylabel="Patient-macro MAE (mV)" if key != "mse" else "MSE (standardized units)",
        xlabel="Epoch",
        title=run.name,
    )
    ax.legend()
    if contrastive:
        axes[0, 1].plot(history.epoch, history.contrastive_loss, label="Auxiliary NT-Xent")
        axes[0, 1].set(xlabel="Epoch", ylabel="Contrastive loss")
        axes[0, 1].legend()
    save_figure(fig, out / "learning_curves.png")


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


def authorize_split(cfg, split, run=None):
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if split == "test":
        settings = cfg.get("evaluation", {})
        if not settings.get("allow_test", False) or cfg.get("subset", {}).get("mode") != "full":
            raise ValueError("Fold 10 is reserved: require full cohort and explicit allow_test")
        selection = read_json(settings["selection_path"])
        if (
            selection["fingerprint"]
            != read_json(Path(cfg["data"]["cache"]) / "manifest.json")["fingerprint"]
        ):
            raise ValueError("Final selection cohort mismatch")
        if run is not None and run.name not in ("persistence", "repeat_window", "training_center"):
            if selection["checkpoints"].get(run.name) != file_hash(run / "best.pt"):
                raise ValueError("Checkpoint is not in the frozen validation selection")


def arrays_evaluation(cfg, run, split, pred, true, observed, meta, manifest):
    import matplotlib.pyplot as plt

    out = Path(cfg["paths"]["outputs"]) / run.name / split
    out.mkdir(parents=True, exist_ok=True)
    summary = metric_summary(pred, true, meta.patient_id.to_numpy())
    summary.update(
        split=split,
        fingerprint=manifest["fingerprint"],
        comparison_key={k: manifest["protocol"][k] for k in ("data", "subset")},
    )
    qrs, peaks, detections, timing = qrs_metrics(
        pred,
        true,
        observed,
        cfg["data"]["sampling_rate"],
        cfg["evaluation"].get("peak_tolerance_ms", 100),
    )
    summary.update(qrs)
    peaks.to_csv(out / "peak_metrics.csv", index=False)
    per_record = meta.copy()
    error = pred.astype(float) - true.astype(float)
    per_record["mae_mV"] = np.abs(error).mean(axis=(1, 2))
    per_record["mse_mV2"] = (error**2).mean(axis=(1, 2))
    per_record.to_csv(run / f"{split}_per_record_metrics.csv", index=False)
    subgroup_metrics(pred, true, meta).to_csv(out / "subgroup_metrics.csv", index=False)
    pd.DataFrame(
        {
            "lead": manifest["protocol"]["leads"],
            "mae_mV": summary["mae_by_lead"],
            "rmse_mV": summary["rmse_by_lead"],
            "pearson": summary["pearson_by_lead"],
        }
    ).to_csv(out / "per_lead_metrics.csv", index=False)
    pd.DataFrame(
        {
            "seconds": np.arange(pred.shape[-1]) / cfg["data"]["sampling_rate"],
            "mae_mV": np.abs(error).mean(axis=(0, 1)),
            "rmse_mV": np.sqrt((error**2).mean(axis=(0, 1))),
        }
    ).to_csv(out / "horizon_metrics.csv", index=False)
    metric_plots(out)
    summary["example_ecg_ids"] = plot_examples(observed, pred, true, meta, out, cfg)
    example = int(np.argsort(per_record.mae_mV.to_numpy(), kind="stable")[len(meta) // 2])
    fs = cfg["data"]["sampling_rate"]
    t = np.arange(pred.shape[-1]) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax in axes:
        ax.plot(t, true[example, 1], label="Actual lead II")
        ax.plot(t, pred[example, 1], label="Predicted lead II")
        for values, color, label in zip(
            detections[example], ["black", "red"], ["Actual R", "Predicted R"], strict=True
        ):
            voltage = true if color == "black" else pred
            ax.scatter(t[values], voltage[example, 1, values], c=color, marker="x", label=label)
        ax.legend(fontsize=8)
        ax.set(xlabel="Forecast time (s)", ylabel="mV")
    reference = detections[example][0]
    if len(reference):
        center = reference[len(reference) // 2] / fs
        axes[1].set_xlim(max(0, center - 0.25), min(pred.shape[-1] / fs, center + 0.25))
    axes[0].set_title(
        f"Median-error example ECG {int(meta.iloc[example].ecg_id)}; detector-derived peaks"
    )
    save_figure(fig, out / "qrs_overlay_zoom.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(timing, bins=20)
    ax.set(
        xlabel="Matched peak timing error (ms)",
        ylabel="Peaks",
        title="No matched peaks" if not timing else "Matched R-peak timing errors",
    )
    save_figure(fig, out / "peak_timing_errors.png")
    np.savez_compressed(
        run / f"{split}_predictions.npz",
        pred=pred,
        true=true,
        ecg_id=meta.ecg_id.to_numpy(),
        patient_id=meta.patient_id.to_numpy(),
    )
    write_json(run / f"{split}_metrics.json", summary)
    return summary


def metric_plots(out):
    """Render already-saved metrics without repeating inference or selecting examples."""
    import matplotlib.pyplot as plt

    for filename, index, title in (
        ("per_lead_metrics", "lead", "Errors by ECG lead"),
        ("horizon_metrics", "seconds", "Errors through the forecast horizon"),
        ("subgroup_metrics", "group", "Errors by overlapping patient subgroups"),
    ):
        frame = pd.read_csv(out / f"{filename}.csv")
        fig, ax = plt.subplots(figsize=(11, 5))
        values = frame.set_index(index)[["mae_mV", "rmse_mV"]]
        if filename == "horizon_metrics":
            values.plot(ax=ax)
        else:
            values.plot.bar(ax=ax)
            ax.tick_params(axis="x", rotation=40)
        ax.set(ylabel="Error (mV)", title=title)
        save_figure(fig, out / f"{filename}.png")


def evaluate_run(cfg, architecture, variant, split="val"):
    manifest = validate_cache(cfg)
    run = run_directory(cfg, architecture, variant)
    authorize_split(cfg, split, run)
    if not (run / "best.pt").exists() or not (run / "config.json").exists():
        raise ValueError(f"No selected checkpoint at {run}; run train.py first")
    recorded = read_json(run / "config.json")
    if recorded["architecture"] != architecture or recorded["variant"] != variant:
        raise ValueError("Requested model/variant differs from the saved run")
    if recorded["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Evaluation cache differs from training")
    loader = make_loaders(cfg, variant, splits=(split,))[split]
    if loader.dataset.selected_features != recorded["selected_features"]:
        raise ValueError("Selected feature order differs from training")
    device = runtime(cfg)
    model = build_model(recorded["config"], architecture, loader.dataset.static_dim).to(device)
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    pred, true = predict(model, loader, device)
    pred, true = [physical_arrays(a, manifest, cfg["data"].get("layout")) for a in (pred, true)]
    observed = np.load(Path(cfg["data"]["cache"]) / f"raw_X_{split}.npy")
    summary = arrays_evaluation(
        cfg, run, split, pred, true, observed, loader.dataset.metadata, manifest
    )
    summary.update(best_epoch=checkpoint["epoch"], checkpoint_sha256=file_hash(run / "best.pt"))
    write_json(run / f"{split}_metrics.json", summary)
    print(f"{run.name} {split}: patient-macro MAE={summary['patient_macro_mae']:.6f} mV", flush=True)
    return summary


def baseline_runs(cfg, split="val"):
    manifest = validate_cache(cfg)
    authorize_split(cfg, split)
    ds = ECGDataset(cfg, split, "ecg_only")
    observed = np.load(Path(cfg["data"]["cache"]) / f"raw_X_{split}.npy")
    true = physical_arrays(np.asarray(ds.y), manifest)
    horizon = cfg["data"]["horizon"]
    predictions = {
        "persistence": np.repeat(observed[:, :, -1:], horizon, axis=2),
        "repeat_window": np.tile(
            observed[:, :, -min(horizon, observed.shape[-1]) :],
            (1, 1, int(np.ceil(horizon / min(horizon, observed.shape[-1])))),
        )[:, :, :horizon],
    }
    # A train-only constant reference exposes apparent gains from near-flat outputs.
    training_raw = np.load(Path(cfg["data"]["cache"]) / "raw_X_train.npy", mmap_mode="r")
    center = np.array([np.median(training_raw[:, j, :]) for j in range(12)])
    predictions["training_center"] = np.broadcast_to(center[None, :, None], true.shape).copy()
    for name, pred in predictions.items():
        run = Path(cfg["paths"]["experiments"]) / name
        run.mkdir(parents=True, exist_ok=True)
        write_json(
            run / "config.json",
            {
                "config": cfg,
                "baseline": name,
                "definition": "constant per-lead training-input median"
                if name == "training_center"
                else name,
            },
        )
        arrays_evaluation(cfg, run, split, pred, true, observed, ds.metadata, manifest)


def freeze_selection(cfg, names):
    manifest = validate_cache(cfg)
    checkpoints = {}
    for name in names:
        run = Path(cfg["paths"]["experiments"]) / name
        metrics = read_json(run / "val_metrics.json")
        if metrics["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Selection cohort mismatch")
        checkpoints[name] = file_hash(run / "best.pt")
    path = Path(cfg["paths"]["experiments"]) / "selection.json"
    if path.exists():
        raise ValueError("Selection already frozen; use a new experiment directory")
    write_json(
        path,
        {
            "fingerprint": manifest["fingerprint"],
            "checkpoints": checkpoints,
            "objective": "validation patient-macro MAE (mV)",
            "used_test": False,
        },
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--model", "-model", choices=["lstm", "cnn_lstm"])
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--baselines", action="store_true", help="Evaluate all three simple forecasts")
    parser.add_argument(
        "--freeze", nargs="+", metavar="RUN",
        help="Freeze validation-selected checkpoints before test inference",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args.experiment)
    if args.device:
        cfg["training"]["device"] = args.device
    with FileLock(".pipeline.lock", timeout=0):
        if args.freeze:
            print(freeze_selection(cfg, args.freeze))
        elif args.baselines:
            baseline_runs(cfg, args.split)
        else:
            evaluate_run(
                cfg,
                args.model or cfg.get("experiment", {}).get("architecture", "lstm"),
                args.variant or cfg.get("experiment", {}).get("variant", "patient_history"),
                args.split,
            )
