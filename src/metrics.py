"""Physical-mV errors and paired uncertainty with equal patient weighting."""

import numpy as np
import pandas as pd

from config import LEADS


def metric_summary(prediction, target, patients):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 3 or p.shape[-1] != 12 or not len(p):
        raise ValueError("Expected nonempty matching [record,time,12] arrays")
    if len(patients) != len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Invalid patient count or nonfinite metric inputs")
    error = p - y
    per_record = pd.DataFrame(
        {
            "patient_id": patients,
            "mae_mV": np.abs(error).mean(axis=(1, 2)),
            "mse_mV2": np.square(error).mean(axis=(1, 2)),
        }
    )
    mse = float(np.square(error).mean())
    centered_p, centered_y = p - p.mean(axis=(0, 1)), y - y.mean(axis=(0, 1))
    denominator = np.sqrt(
        np.square(centered_p).sum(axis=(0, 1)) * np.square(centered_y).sum(axis=(0, 1))
    )
    correlation = np.divide(
        (centered_p * centered_y).sum(axis=(0, 1)),
        denominator,
        out=np.full(12, np.nan),
        where=denominator > 1e-12,
    )
    ss = float(np.square(y - y.mean()).sum())
    by_lead = pd.DataFrame(
        {
            "lead": LEADS,
            "mae_mV": np.abs(error).mean(axis=(0, 1)),
            "rmse_mV": np.sqrt(np.square(error).mean(axis=(0, 1))),
            "pearson": correlation,
        }
    )
    summary = {
        "mae_mV": float(per_record.mae_mV.mean()),
        "mse_mV2": mse,
        "rmse_mV": float(np.sqrt(mse)),
        "patient_macro_mae_mV": float(per_record.groupby("patient_id").mae_mV.mean().mean()),
        "r2": float(1 - np.square(error).sum() / ss) if ss > 1e-12 else None,
        "pearson_mean": float(np.nanmean(correlation)) if np.isfinite(correlation).any() else None,
        "records": len(p),
        "patients": int(per_record.patient_id.nunique()),
    }
    return summary, by_lead, per_record


def bootstrap_pair(first, second, samples=1000, seed=42):
    if set(first.ecg_id) != set(second.ecg_id):
        raise ValueError("Cannot compare different held-out record sets")
    joined = first[["ecg_id", "patient_id", "mae_mV"]].merge(
        second[["ecg_id", "patient_id", "mae_mV"]],
        on=["ecg_id", "patient_id"],
        validate="one_to_one",
        suffixes=("_a", "_b"),
    )
    if len(joined) != len(first):
        raise ValueError("Patient IDs differ between comparisons")
    joined["difference"] = joined.mae_mV_b - joined.mae_mV_a
    delta = joined.groupby("patient_id").difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    boot = [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(samples)]
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "difference_B_minus_A_mV": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "patients": len(delta),
        "bootstrap_samples": samples,
        "conclusion": "helped" if high < 0 else ("did not help" if low > 0 else "inconclusive"),
    }
