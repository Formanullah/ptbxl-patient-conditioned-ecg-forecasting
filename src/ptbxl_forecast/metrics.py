"""Forecast evaluation metrics (Phase 6). All per-record; aggregate with ``stats``.

Shapes: ``history`` (n, H, 12), ``y`` / ``yhat`` (n, F, 12), raw mV at 100 Hz.

Beat-based metrics detect R peaks on lead II of ``[true history | segment]`` and use
only beats fully inside the forecast window. Templates are the median beat from
-250 to +450 ms around R. Amplitudes are measured relative to the PR level at R - 80 ms,
so morphology is scored independently of the baseline level (which pointwise metrics
already cover). A forecast without detectable beats gets a flat template.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

FS = 100
PRE, POST = 25, 45  # template window around R (samples)
PR, ST, T_LO, T_HI = -8, 8, 15, 40  # offsets from R (samples): PR level, ST point, T window
MATCH_TOL = 5  # beat matching tolerance (samples = 50 ms)
DTW_BAND = 10  # Sakoe-Chiba half-width (samples = 100 ms)
# Interpretive bands for the per-record beat-timing error (D6.3/D6.6; reporting only; the
# binding comparison rule is relative, D6.2):
#   <= 25 ms  within measurement noise (detector jitter on a perfect-timing forecast)
#   <= 50 ms  acceptable: beats match under the +-50 ms tolerance
#   <= 150 ms misaligned (150 ms = AAMI EC57 beat-matching window)
#   >  150 ms rhythm failure; 300 ms = no beats (cap)
BEAT_TIMING_BANDS_MS = (25, 50, 150)
INDEPENDENT_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1-V6 (Phase 2, D2.4)


# ------------------------------------------------------------------ pointwise
def pointwise(y: np.ndarray, yhat: np.ndarray) -> pd.DataFrame:
    err = yhat - y
    mse = (err**2).mean(axis=(1, 2))
    yc, pc = y - y.mean(1, keepdims=True), yhat - yhat.mean(1, keepdims=True)
    den = np.sqrt((yc**2).sum(1) * (pc**2).sum(1))
    # A (numerically) constant forecast or target lead has no defined correlation: score 0.
    ok = ((pc**2).sum(1) > 1e-12) & ((yc**2).sum(1) > 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ok, (yc * pc).sum(1) / np.where(ok, den, 1), 0.0)
        snr = 10 * np.log10((y**2).sum(axis=(1, 2)) / np.maximum((err**2).sum(axis=(1, 2)), 1e-12))
    return pd.DataFrame(
        {"mae": np.abs(err).mean(axis=(1, 2)), "mse": mse, "rmse": np.sqrt(mse),
         "pearson": r.mean(1), "snr_db": snr,
         "mae_8lead": np.abs(err[..., INDEPENDENT_LEADS]).mean(axis=(1, 2))}
    )  # fmt: skip


# ------------------------------------------------------------------ DTW
def _dtw(a: np.ndarray, b: np.ndarray, w: int) -> float:
    n = len(a)
    prev = np.full(n + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(n + 1, np.inf)
        for j in range(max(1, i - w), min(n, i + w) + 1):
            cur[j] = abs(a[i - 1] - b[j - 1]) + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[n] / n


try:
    from numba import njit

    _dtw = njit(cache=True)(_dtw)
except ImportError:  # pragma: no cover
    pass


def dtw_mae(y: np.ndarray, yhat: np.ndarray, band: int = DTW_BAND) -> np.ndarray:
    """Banded DTW with |.| cost, divided by length (mV), averaged over leads."""
    y64, p64 = y.astype(np.float64), yhat.astype(np.float64)
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = np.mean([_dtw(y64[i, :, j], p64[i, :, j], band) for j in range(y.shape[2])])
    return out


# ------------------------------------------------------------------ beats
def detect_beats(history: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """R peaks (indices within ``segment``) detected on lead II of [history | segment]."""
    import neurokit2 as nk

    x = np.concatenate([history[:, 1], segment[:, 1]]).astype(np.float64)
    if np.ptp(segment[:, 1]) < 1e-6:
        return np.array([], dtype=int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_peaks(nk.ecg_clean(x, sampling_rate=FS), sampling_rate=FS)
        p = np.asarray(info["ECG_R_Peaks"], dtype=int) - len(history)
    except Exception:
        return np.array([], dtype=int)
    return p[p >= 0]


def template(segment: np.ndarray, peaks: np.ndarray) -> np.ndarray | None:
    """Median beat (PRE + POST, 12), PR-level referenced; None if no full beat."""
    peaks = peaks[(peaks >= PRE) & (peaks + POST <= len(segment))]
    if len(peaks) == 0:
        return None
    t = np.median(np.stack([segment[p - PRE : p + POST] for p in peaks]), axis=0)
    return t - t[PRE + PR]


def amplitudes(t: np.ndarray) -> np.ndarray:
    """(3, 12): R amplitude (signed max-|.| within R +- 3), ST level, T-wave extreme."""
    r_win = t[PRE - 3 : PRE + 4]
    r = r_win[np.abs(r_win).argmax(0), np.arange(t.shape[1])]
    t_win = t[PRE + T_LO : PRE + T_HI]
    tw = t_win[np.abs(t_win).argmax(0), np.arange(t.shape[1])]
    return np.stack([r, t[PRE + ST], tw])


def beat_timing_ms(pred: np.ndarray, true: np.ndarray, cap: int = 30) -> float:
    """Symmetric mean distance (ms) from each beat to the nearest beat of the other
    sequence, capped at ``cap`` samples (300 ms). Defined for every forecast: no
    predicted beats (e.g. a flat line) scores the cap. Added post hoc in Phase 6."""
    if len(pred) == 0 and len(true) == 0:
        return 0.0
    if len(pred) == 0 or len(true) == 0:
        return cap * 1000 / FS
    d = np.abs(true[:, None] - pred[None, :])
    both = (np.minimum(d.min(1), cap).mean() + np.minimum(d.min(0), cap).mean()) / 2
    return float(both * 1000 / FS)


def beat_f1(pred: np.ndarray, true: np.ndarray, tol: int = MATCH_TOL) -> float:
    if len(pred) == 0 and len(true) == 0:
        return 1.0
    if len(pred) == 0 or len(true) == 0:
        return 0.0
    used, tp = set(), 0
    for p in pred:
        d = np.abs(true - p)
        for k in np.argsort(d):
            if d[k] > tol:
                break
            if k not in used:
                used.add(k)
                tp += 1
                break
    return 2 * tp / (len(pred) + len(true))


def beat_metrics(
    history: np.ndarray, y: np.ndarray, yhat: np.ndarray, true_peaks: list[np.ndarray] | None = None
) -> pd.DataFrame:
    """Morphology and rhythm metrics. ``true_peaks`` may be precomputed (reused across models)."""
    rows = []
    for i in range(len(y)):
        tp = true_peaks[i] if true_peaks is not None else detect_beats(history[i], y[i])
        pp = detect_beats(history[i], yhat[i])
        tt, pt = template(y[i], tp), template(yhat[i], pp)
        row = {"beat_f1": beat_f1(pp, tp), "beat_timing_ms": beat_timing_ms(pp, tp),
               "n_pred_beats": len(pp), "n_true_beats": len(tp)}  # fmt: skip
        if tt is not None:
            pt = np.zeros_like(tt) if pt is None else pt
            a_t, a_p = amplitudes(tt), amplitudes(pt)
            row.update(template_mae=float(np.abs(pt - tt).mean()), r_amp_err=float(np.abs(a_p[0] - a_t[0]).mean()),
                       st_err=float(np.abs(a_p[1] - a_t[1]).mean()), t_amp_err=float(np.abs(a_p[2] - a_t[2]).mean()))  # fmt: skip
        if len(tp) >= 2 and len(pp) >= 2:
            row["rr_err_ms"] = float(abs(np.diff(pp).mean() - np.diff(tp).mean()) * 1000 / FS)
        rows.append(row)
    cols = [
        "template_mae",
        "r_amp_err",
        "st_err",
        "t_amp_err",
        "beat_f1",
        "beat_timing_ms",
        "rr_err_ms",
        "n_pred_beats",
        "n_true_beats",
    ]
    return pd.DataFrame(rows).reindex(columns=cols)


def evaluate(
    history: np.ndarray, y: np.ndarray, yhat: np.ndarray, true_peaks: list[np.ndarray] | None = None,
    dtw: bool = True, beats: bool = True,
) -> pd.DataFrame:  # fmt: skip
    """All metrics, one row per record."""
    parts = [pointwise(y, yhat)]
    if dtw:
        parts.append(pd.DataFrame({"dtw_mae": dtw_mae(y, yhat)}))
    if beats:
        parts.append(beat_metrics(history, y, yhat, true_peaks))
    return pd.concat(parts, axis=1)
