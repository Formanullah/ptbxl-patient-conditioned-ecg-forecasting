"""Physical-unit metrics, subgroup summaries, and reproducible visualizations."""

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

from config import CLASSES, LEADS, VARIANTS, load_config, read_json, runtime, write_json
from models import build_model
from preprocessing import ECGDataset, make_loaders, validate_cache


def metric_summary(pred, true, patients):
    if pred.shape != true.shape or pred.ndim != 3 or pred.shape[1] != 12:
        raise ValueError("Expected matching N x 12 x horizon arrays")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Nonfinite predictions or targets")
    pred, true = pred.astype(np.float64), true.astype(np.float64)
    error = pred - true
    mse = float(np.mean(error**2))
    ss = float(np.sum((true - true.mean()) ** 2))
    p, t = [a.transpose(1, 0, 2).reshape(12, -1) for a in (pred, true)]
    p, t = p - p.mean(1, keepdims=True), t - t.mean(1, keepdims=True)
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    corr = np.divide((p * t).sum(1), denom, out=np.full(12, np.nan), where=denom > 1e-12)
    per_record = np.mean(np.abs(error), axis=(1, 2))
    patient_mae = (
        pd.DataFrame({"patient": patients, "mae": per_record}).groupby("patient").mae.mean()
    )
    return {
        "mse": mse,
        "mae": float(per_record.mean()),
        "rmse": float(np.sqrt(mse)),
        "r2": float(1 - np.sum(error**2) / ss) if ss > 1e-12 else None,
        "pearson_mean": float(np.nanmean(corr)) if np.isfinite(corr).any() else None,
        "pearson_by_lead": [float(v) if np.isfinite(v) else None for v in corr],
        "undefined_pearson_leads": int((~np.isfinite(corr)).sum()),
        "mae_by_lead": np.abs(error).mean(axis=(0, 2)).tolist(),
        "rmse_by_lead": np.sqrt((error**2).mean(axis=(0, 2))).tolist(),
        "patient_macro_mae": float(patient_mae.mean()),
        "n_test_records": len(true),
        "n_test_patients": len(patient_mae),
    }


def subgroup_metrics(pred, true, metadata):
    age = pd.to_numeric(metadata.age, errors="coerce")
    age_group = pd.cut(
        age,
        [-np.inf, 40, 60, 80, np.inf],
        right=False,
        labels=["under_40", "40_59", "60_79", "80_plus"],
    ).astype(object)
    age_group = age_group.fillna("unknown")
    sex_group = metadata.sex.map({0: "male", 1: "female"}).fillna("unknown")
    groups = [
        ("age", label, age_group.eq(label).to_numpy())
        for label in ["under_40", "40_59", "60_79", "80_plus", "unknown"]
    ]
    groups += [
        ("sex", label, sex_group.eq(label).to_numpy()) for label in ["male", "female", "unknown"]
    ]
    groups += [("diagnosis", label, metadata[label].eq(1).to_numpy()) for label in CLASSES]
    groups += [
        ("history", label, metadata.history_available.eq(value).to_numpy())
        for label, value in [("available", 1), ("missing", 0)]
    ]
    error = pred.astype(np.float64) - true.astype(np.float64)
    rows = []
    for kind, label, mask in groups:
        rows.append(
            {
                "category": kind,
                "group": label,
                "records": int(mask.sum()),
                "patients": int(metadata.loc[mask, "patient_id"].nunique()),
                "mae_mV": float(np.abs(error[mask]).mean()) if mask.any() else None,
                "rmse_mV": float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None,
            }
        )
    return pd.DataFrame(rows)


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


def evaluate_arrays(pred, true, x, metadata, run, cfg, manifest):
    run = Path(run)
    out = Path(cfg["paths"]["outputs"]) / run.name
    out.mkdir(parents=True, exist_ok=True)
    summary = metric_summary(pred, true, metadata.patient_id.to_numpy())
    summary["fingerprint"] = manifest["fingerprint"]
    np.savez_compressed(
        run / "predictions.npz",
        pred=pred,
        true=true,
        ecg_id=metadata.ecg_id.to_numpy(),
        patient_id=metadata.patient_id.to_numpy(),
    )
    errors = pred.astype(np.float64) - true.astype(np.float64)
    record_metrics = metadata.copy()
    record_metrics["mae_mV"] = np.abs(errors).mean(axis=(1, 2))
    record_metrics["mse_mV2"] = (errors**2).mean(axis=(1, 2))
    record_metrics.to_csv(run / "per_record_metrics.csv", index=False)
    groups = subgroup_metrics(pred, true, metadata)
    groups.to_csv(out / "subgroup_metrics.csv", index=False)
    per_lead = pd.DataFrame(
        {
            "lead": LEADS,
            "mae_mV": summary["mae_by_lead"],
            "rmse_mV": summary["rmse_by_lead"],
            "pearson": summary["pearson_by_lead"],
        }
    )
    per_lead.to_csv(out / "per_lead_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    per_lead.set_index("lead")[["mae_mV", "rmse_mV"]].plot.bar(ax=ax)
    ax.set(title=run.name, ylabel="Error (mV)")
    save_figure(fig, out / "per_lead_errors.png")
    horizon = pd.DataFrame(
        {
            "seconds": np.arange(true.shape[-1]) / cfg["data"]["sampling_rate"],
            "mae_mV": np.abs(errors).mean(axis=(0, 1)),
            "rmse_mV": np.sqrt((errors**2).mean(axis=(0, 1))),
        }
    )
    horizon.to_csv(out / "horizon_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(horizon.seconds, horizon.mae_mV, label="MAE")
    ax.plot(horizon.seconds, horizon.rmse_mV, label="RMSE")
    ax.set(xlabel="Seconds into forecast", ylabel="Error (mV)", title=run.name)
    ax.legend()
    save_figure(fig, out / "horizon_error.png")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, category in zip(axes, ["age", "sex", "diagnosis", "history"], strict=True):
        group = groups.loc[(groups.category == category) & (groups.records > 0)]
        ax.bar(group.group, group.mae_mV)
        ax.set(title=category, ylabel="MAE (mV)")
        ax.tick_params(axis="x", rotation=45)
    save_figure(fig, out / "subgroup_errors.png")
    summary["example_ecg_ids"] = plot_examples(x, pred, true, metadata, out, cfg)
    write_json(run / "test_metrics.json", summary)
    return summary


def learning_curves(run, cfg):
    run = Path(run)
    history = pd.read_csv(run / "history.csv")
    out = Path(cfg["paths"]["outputs"]) / run.name
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.epoch, history.train_mse, label="Train forecast MSE")
    axes[0].plot(history.epoch, history.val_mse, label="Validation forecast MSE")
    axes[0].set(ylabel="MSE (standardized units)", xlabel="Epoch", title=run.name)
    axes[0].legend()
    axes[1].plot(history.epoch, history.contrastive_loss, label="Auxiliary NT-Xent")
    axes[1].set(xlabel="Epoch", ylabel="Contrastive loss")
    axes[1].legend()
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
        write_json(run / "config.json", {"config": cfg, "smoke": False, "baseline": name})
        result = evaluate_arrays(pred, true, x, ds.metadata, run, cfg, manifest)
        print(f"BASELINE {name}: MAE={result['mae']:.6f} mV", flush=True)


def evaluate_saved(cfg, architecture, variant):
    """Evaluate an existing selected checkpoint without training or changing weights."""
    manifest = validate_cache(cfg)
    run = Path(cfg["paths"]["experiments"]) / f"{architecture}_{variant}"
    if not (run / "best.pt").exists() or not (run / "config.json").exists():
        raise ValueError(f"No selected checkpoint at {run}; run train first")
    recorded = read_json(run / "config.json")
    if recorded["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Evaluation cache differs from training cache")
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
            "parameters_used_by_loss": recorded.get(
                "parameters_used_by_loss", recorded["parameters"]
            ),
            "architecture": architecture,
            "variant": variant,
            "smoke": False,
        }
    )
    write_json(run / "test_metrics.json", summary)
    learning_curves(run, cfg)
    print(f"EVALUATED {run.name}: MAE={summary['mae']:.6f} mV, RMSE={summary['rmse']:.6f} mV")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--model", "-model", choices=["lstm", "cnn_lstm"], default="lstm")
    parser.add_argument("--variant", choices=VARIANTS, default="patient_history")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--baselines", action="store_true", help="Evaluate the two simple forecasts"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device

    with FileLock(".pipeline.lock", timeout=0):
        if args.baselines:
            baselines(config)
        else:
            evaluate_saved(config, args.model, args.variant)
