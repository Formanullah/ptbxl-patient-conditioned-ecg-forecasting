"""Pure physical-unit metrics and paired patient-bootstrap estimates."""

import numpy as np
import pandas as pd

from config import CLASSES

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
        "n_records": len(true),
        "n_patients": len(patient_mae),
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


def physical_arrays(array, manifest, layout="channels_first"):
    if layout == "time_first":
        array = array.transpose(0, 2, 1)
    scale, center = [
        np.asarray(manifest[k], dtype=np.float32)[None, :, None] for k in ("std_mV", "mean_mV")
    ]
    return array * scale + center


def bootstrap_pair(a, b, samples=1000, seed=42):
    """Paired patient-macro MAE delta B minus A; negative favors B."""
    if set(a.ecg_id) != set(b.ecg_id):
        raise ValueError("Cannot pair different held-out record sets")
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
