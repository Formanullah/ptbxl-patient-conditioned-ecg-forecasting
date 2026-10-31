"""Physical-unit metrics, subgroup summaries, and reproducible visualizations."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLASSES, LEADS, read_json, write_json


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


def bootstrap_pair(a, b, samples=1000, seed=42):
    """Paired patient-macro MAE delta B minus A; negative favors B."""
    if set(a.ecg_id) != set(b.ecg_id):
        raise ValueError("Cannot pair different test record sets")
    joined = a[["ecg_id", "patient_id", "mae_mV"]].merge(
        b[["ecg_id", "patient_id", "mae_mV"]],
        on="ecg_id",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if not joined.patient_id_a.equals(joined.patient_id_b):
        raise ValueError("Patient assignments differ between runs")
    joined["delta"] = joined.mae_mV_b - joined.mae_mV_a
    delta = joined.groupby("patient_id_a").delta.mean().to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.empty(samples)
    for i in range(samples):
        boots[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    return {
        "delta_patient_macro_mae_mV": float(delta.mean()),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "patients": len(delta),
        "bootstrap_samples": samples,
    }


def compare_runs(cfg):
    root = Path(cfg["paths"]["experiments"])
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    rows, fingerprints = [], set()
    for run in sorted(root.iterdir()):
        if not (run / "test_metrics.json").exists():
            continue
        metrics, config = read_json(run / "test_metrics.json"), read_json(run / "config.json")
        if config.get("smoke"):
            continue
        fingerprints.add(metrics["fingerprint"])
        rows.append(
            {
                "run": run.name,
                **{
                    k: metrics[k]
                    for k in (
                        "mae",
                        "rmse",
                        "mse",
                        "r2",
                        "pearson_mean",
                        "patient_macro_mae",
                        "n_test_records",
                        "n_test_patients",
                    )
                },
                "epochs": metrics.get("epochs_completed", 0),
                "best_epoch": metrics.get("best_epoch"),
                "training_seconds": metrics.get("training_seconds", 0),
                "parameters": metrics.get("parameters", 0),
            }
        )
    if len(fingerprints) > 1:
        raise ValueError("Refusing comparison of mismatched protocols/caches")
    if not rows:
        raise ValueError("No measured full-data results found")
    table = pd.DataFrame(rows)
    table.to_csv(out / "model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    display = table.copy()
    display["label"] = display.run.map(
        lambda name: (
            name.replace("cnn_lstm_", "CNN-LSTM / ")
            .replace("lstm_", "LSTM / ")
            .replace("patient_history_contrastive", "History + contrastive")
            .replace("patient_history", "History")
            .replace("ecg_only", "ECG only")
            .replace("demographics", "Demographics")
            .replace("repeat_window", "Repeated window")
            .replace("persistence", "Persistence")
        )
    )
    display.set_index("label")[["mae", "rmse"]].rename(
        columns={"mae": "MAE", "rmse": "RMSE"}
    ).plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set(
        xlabel="Error (mV)",
        ylabel="",
        title="Small-data pilot: held-out patients"
        if cfg.get("subset")
        else "Held-out patient test fold",
    )
    save_figure(fig, out / "model_comparison.png")
    names = set(table.run)
    pairs = []
    for architecture in ("lstm", "cnn_lstm"):
        pairs += [
            (f"{architecture}_ecg_only", f"{architecture}_demographics", "demographics"),
            (f"{architecture}_demographics", f"{architecture}_patient_history", "history"),
            (
                f"{architecture}_patient_history",
                f"{architecture}_patient_history_contrastive",
                "contrastive",
            ),
        ]
    pairs += [
        (f"lstm_{v}", f"cnn_lstm_{v}", "architecture")
        for v in ("ecg_only", "demographics", "patient_history", "patient_history_contrastive")
    ]
    comparisons = []
    for a, b, effect in pairs:
        if a in names and b in names:
            result = bootstrap_pair(
                pd.read_csv(root / a / "per_record_metrics.csv"),
                pd.read_csv(root / b / "per_record_metrics.csv"),
                cfg["evaluation"]["bootstrap_samples"],
                cfg["training"]["seed"],
            )
            comparisons.append({"effect": effect, "A": a, "B": b, **result})
    pd.DataFrame(comparisons).to_csv(out / "paired_patient_bootstrap.csv", index=False)
    return table
