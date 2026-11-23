"""Patient-level aggregation and patient-cluster bootstrap (used by every phase).

Records from the same patient are not independent (23% of records come from repeat
patients, Phase 2), so averages are patient-macro and intervals resample patients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def patient_macro(values, patients) -> float:
    """Mean over patients of each patient's mean (NaNs ignored)."""
    s = pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(patients)).mean()
    return float(s.mean())


def cluster_bootstrap(values, patients, n_boot: int = 1000, seed: int = 0, ci: float = 0.95):
    """Patient-macro mean and percentile CI, resampling patients with replacement."""
    per_patient = (
        pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(patients)).mean().dropna()
    )
    v = per_patient.to_numpy()
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boots, [50 * (1 - ci), 50 * (1 + ci)])
    return float(v.mean()), float(lo), float(hi)


def paired_effect(a, b, patients) -> float:
    """Standardized paired difference (mean / SD) of per-patient means of a - b."""
    d = (
        pd.Series(np.asarray(a, float) - np.asarray(b, float))
        .groupby(np.asarray(patients))
        .mean()
        .dropna()
    )
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0


PRIMARY = "mae"
GATEKEEPERS = ("dtw_mae", "template_mae", "beat_timing_ms")  # all lower-is-better (D6.2, D6.3)


def paired_diff_ci(a, b, patients, n_boot: int = 1000, seed: int = 0):
    """Patient-macro mean of a - b with a patient-cluster bootstrap 95% CI."""
    return cluster_bootstrap(np.asarray(a, float) - np.asarray(b, float), patients, n_boot, seed)


def compare(
    candidate: pd.DataFrame, reference: pd.DataFrame, patients, n_boot: int = 1000, seed: int = 0
) -> dict:
    """D6.2 rule: the candidate beats the reference iff the ΔMAE CI is entirely below 0
    and no gatekeeper's Δ CI is entirely above 0 (worse). Rows must be the same records."""
    out: dict = {}
    for m in (PRIMARY, *GATEKEEPERS):
        d, lo, hi = paired_diff_ci(candidate[m], reference[m], patients, n_boot, seed)
        out[m] = {"diff": d, "lo": lo, "hi": hi}
    out["mae_better"] = bool(out[PRIMARY]["hi"] < 0)
    out["gatekeepers_not_worse"] = all(out[g]["lo"] <= 0 for g in GATEKEEPERS)
    out["beats_reference"] = bool(out["mae_better"] and out["gatekeepers_not_worse"])
    return out


def _patient_runs(runs, metric: str, codes: np.ndarray, n_patients: int) -> np.ndarray:
    """(runs, patients) matrix of per-patient means of ``metric`` (NaN ignored)."""
    return np.stack([
        pd.Series(np.asarray(r[metric], dtype=float)).groupby(codes).mean()
        .reindex(range(n_patients)).to_numpy()
        for r in runs
    ])  # fmt: skip


def compare_runs(
    candidate_runs: list[pd.DataFrame], reference_runs: list[pd.DataFrame], patients,
    n_boot: int = 1000, seed: int = 0, metrics: tuple[str, ...] = (PRIMARY, *GATEKEEPERS),
) -> dict:  # fmt: skip
    """D8.3 comparison: the D6.2 decision logic with a hierarchical bootstrap that
    resamples training runs within each arm and then patients.

    Each run is a per-record score table of one independently trained model (different
    seeds; calibrated with 6 runs per arm, Phase 8). The statistic is the difference of
    the arms' run-mean patient-macro scores. Patients are paired across arms; patients
    with an undefined metric in any run are dropped for that metric. A deterministic
    reference (e.g. a non-learned baseline) may be passed as a single run.
    """
    if len(candidate_runs) < 2 or len(reference_runs) < 1:
        raise ValueError("the candidate needs >= 2 independently trained runs")
    codes, uniq = pd.factorize(np.asarray(patients))
    rng = np.random.default_rng(seed)
    na, nb = len(candidate_runs), len(reference_runs)
    wa = rng.multinomial(na, [1 / na] * na, size=n_boot) / na
    wb = rng.multinomial(nb, [1 / nb] * nb, size=n_boot) / nb
    counts = rng.multinomial(len(uniq), [1 / len(uniq)] * len(uniq), size=n_boot).astype(float)
    out: dict = {}
    for m in metrics:
        a = _patient_runs(candidate_runs, m, codes, len(uniq))
        b = _patient_runs(reference_runs, m, codes, len(uniq))
        ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
        a, b, c = a[:, ok], b[:, ok], counts[:, ok]
        boots = (c * (wa @ a - wb @ b)).sum(axis=1) / c.sum(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[m] = {"diff": float((a.mean(0) - b.mean(0)).mean()), "lo": float(lo), "hi": float(hi)}
    if PRIMARY in out:
        out["mae_better"] = bool(out[PRIMARY]["hi"] < 0)
        out["gatekeepers_not_worse"] = all(out[g]["lo"] <= 0 for g in GATEKEEPERS if g in out)
        out["beats_reference"] = bool(out["mae_better"] and out["gatekeepers_not_worse"])
    return out
