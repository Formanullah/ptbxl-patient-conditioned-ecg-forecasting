"""Engineered history features (Phase 5 candidates; history only, so causal by construction)."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal as sps
from scipy import stats

FS = 100
PRE, POST = 25, 45


def statistical(x: np.ndarray) -> np.ndarray:
    """Per-lead mean, SD, min, max, skewness, kurtosis: (n, 6 * leads)."""
    feats = [
        x.mean(1),
        x.std(1),
        x.min(1),
        x.max(1),
        stats.skew(x, axis=1),
        stats.kurtosis(x, axis=1),
    ]
    return np.nan_to_num(np.concatenate(feats, axis=1))


def frequency(x: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Log Welch power in ``n_bins`` equal bands over 0-50 Hz, per lead: (n, n_bins * leads)."""
    f, p = sps.welch(x, fs=FS, nperseg=256, axis=1)
    edges = np.linspace(0, 50.01, n_bins + 1)
    bands = [
        p[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    return np.log(np.stack(bands, axis=1) + 1e-8).reshape(len(x), -1)


def _peaks(lead_ii: np.ndarray) -> np.ndarray:
    import neurokit2 as nk

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_peaks(nk.ecg_clean(lead_ii, sampling_rate=FS), sampling_rate=FS)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def morphological_one(x: np.ndarray) -> np.ndarray:
    """Rhythm summary + median beat for one history (T, leads); R peaks from the history only."""
    t_len, leads = x.shape
    p = _peaks(x[:, 1].astype(np.float64))
    rr = np.diff(p) / FS
    rr = rr[(rr > 0.25) & (rr < 2.0)]
    mean_rr = rr.mean() if len(rr) else 0.8
    rhythm = [mean_rr, rr[-1] if len(rr) else mean_rr, rr.std() if len(rr) > 1 else 0.0,
              (t_len - p[-1]) / FS if len(p) else mean_rr / 2, len(p) / (t_len / FS)]  # fmt: skip
    full = p[(p >= PRE) & (p + POST <= t_len)]
    tpl = (
        np.median(np.stack([x[q - PRE : q + POST] for q in full]), 0)
        if len(full)
        else np.zeros((PRE + POST, leads))
    )
    return np.concatenate([rhythm, tpl.ravel()]).astype(np.float32)


def morphological(x: np.ndarray, workers: int = 1) -> np.ndarray:
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            return np.stack(pool.map(morphological_one, list(x), chunksize=64))
    return np.stack([morphological_one(r) for r in x])
