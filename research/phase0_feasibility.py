"""Phase 0: forecasting feasibility study for PTB-XL 100 Hz, 7 s -> 3 s.

Measures how much exploitable structure the ECG itself offers, using only
non-learned per-record forecasts plus oracle diagnostics. Folds 1-9 only; fold 10
is never read. See reports/phase0_feasibility/PREREGISTRATION.md.

    python research/phase0_feasibility.py --data PTB-XL --out reports/phase0_feasibility
"""

from __future__ import annotations

import argparse
import ast
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

FS = 100
N_TOTAL = 1000
H = 700
F = 300
K_CYCLES = 4
PREDICTORS = (
    "zero",
    "mean",
    "median",
    "last",
    "lastsec",
    "sn1",
    "sn4",
    "sn4_trueperiod",
    "oracle",
)
DTW_PREDICTORS = ("zero", "mean", "median", "sn1", "sn4", "sn4_trueperiod", "oracle")
DTW_BAND = 10  # Sakoe-Chiba half-width in samples (100 ms)
SPLITS = (700, 800, 900)
CONTEXTS = (300, 500, 700, 900)  # context lengths for predicting the fixed 9-10 s window
RHYTHM_GROUPS = ("SR", "AFIB", "STACH", "SBRAD", "SARRH", "AFLT", "PACE")
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
QUALITY_FLAGS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems")
SEED = 20261122
N_BOOT = 1000


# ---------------------------------------------------------------- signal helpers
def _dtw_band(a: np.ndarray, b: np.ndarray, w: int) -> float:
    """Banded DTW with |a-b| cost, divided by length so the unit stays mV."""
    n = len(a)
    acc = np.full((n + 1, n + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(n, i + w) + 1):
            acc[i, j] = abs(a[i - 1] - b[j - 1]) + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )
    return acc[n, n] / n


try:
    from numba import njit

    dtw_band = njit(_dtw_band)
except ImportError:  # pragma: no cover - slow pure-Python fallback
    dtw_band = _dtw_band


def rpeaks(sig: np.ndarray) -> np.ndarray:
    import neurokit2 as nk

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.ecg_clean(sig, sampling_rate=FS, method="neurokit")
            _, info = nk.ecg_peaks(clean, sampling_rate=FS, method="neurokit")
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def plausible_rr(peaks: np.ndarray) -> np.ndarray:
    rr = np.diff(peaks)
    return rr[(rr >= 0.25 * FS) & (rr <= 2.0 * FS)]


def estimate_period(hist: np.ndarray, hist_peaks: np.ndarray) -> tuple[float, float]:
    """Fractional cycle length (samples) and normalised ACF height at that lag.

    Uses the multi-lead autocorrelation of the first difference (suppresses
    baseline wander, emphasises QRS), searched near the median history RR.
    """
    d = np.diff(hist, axis=0)
    d = d - d.mean(axis=0)
    n = len(d)
    spec = np.fft.rfft(d, 2 * n, axis=0)
    acf = np.fft.irfft(np.abs(spec) ** 2, axis=0)[:n].sum(axis=1)
    acf = acf / np.arange(n, 0, -1)  # unbiased
    acf = acf / acf[0] if acf[0] > 0 else acf
    rr = plausible_rr(hist_peaks)
    max_lag = min(n // 2, 2 * FS)
    if len(rr) >= 2:
        med = float(np.median(rr))
        lo, hi = int(max(0.25 * FS, 0.75 * med)), int(min(max_lag, np.ceil(1.25 * med)))
    else:
        lo, hi = int(0.25 * FS), max_lag
    if hi <= lo + 1:
        lo, hi = int(0.25 * FS), max_lag
    lag = lo + int(np.argmax(acf[lo : hi + 1]))
    shift = 0.0
    if 0 < lag < len(acf) - 1:
        y0, y1, y2 = acf[lag - 1], acf[lag], acf[lag + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            shift = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return lag + shift, float(acf[lag])


def lerp(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    pos = np.clip(pos, 0, len(x) - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (pos - i0)[:, None]
    return x[i0] * (1 - w) + x[i1] * w


def seasonal(hist: np.ndarray, period: float, horizon: int, k_max: int) -> np.ndarray:
    """Average of up to k_max previous cycles; each is level-matched to the last cycle."""
    n = len(hist)
    t = np.arange(horizon)
    m0 = np.ceil((t + 1) / period)
    preds, levels = [], []
    for k in range(k_max):
        pos = n + t - period * (m0 + k)
        if pos.min() < 0:
            break
        preds.append(lerp(hist, pos))
        start = int(round(n - period * (k + 1)))
        stop = int(round(n - period * k))
        levels.append(hist[max(start, 0) : stop].mean(axis=0))
    if not preds:  # period longer than history: fall back to one wrapped cycle
        return lerp(hist, n - period + (t % period))
    adjusted = [p - lvl for p, lvl in zip(preds, levels, strict=True)]
    return np.mean(adjusted, axis=0) + levels[0]


def oracle_timing(
    sig: np.ndarray, peaks: np.ndarray, hist_len: int, horizon: int, k_max: int
) -> np.ndarray | None:
    """History beat morphology time-warped onto the true future R-R intervals."""
    hp = peaks[peaks < hist_len]
    if len(hp) < 2 or not np.any(peaks >= hist_len):
        return None
    cycles = []
    for i in range(len(hp) - 1, 0, -1):
        a, b = hp[i - 1], hp[i]
        if 0.25 * FS <= b - a <= 2.0 * FS:
            cycles.append((a, b))
        if len(cycles) == k_max:
            break
    if not cycles:
        return None
    t_abs = hist_len + np.arange(horizon)
    idx = np.searchsorted(peaks, t_abs, side="right") - 1
    a = peaks[np.clip(idx, 0, len(peaks) - 1)]
    last_rr = np.diff(peaks)[-1] if len(peaks) > 1 else cycles[0][1] - cycles[0][0]
    nxt = np.where(idx + 1 < len(peaks), peaks[np.clip(idx + 1, 0, len(peaks) - 1)], a + last_rr)
    phase = (t_abs - a) / np.maximum(nxt - a, 1)
    hist = sig[:hist_len]
    preds, levels = [], []
    for a0, b0 in cycles:
        preds.append(lerp(hist, a0 + phase * (b0 - a0)))
        levels.append(hist[a0:b0].mean(axis=0))
    return np.mean([p - lvl for p, lvl in zip(preds, levels, strict=True)], axis=0) + levels[0]


def spectral_features(x: np.ndarray) -> dict[str, float]:
    from scipy.signal import welch

    f, p = welch(x, fs=FS, nperseg=min(512, len(x)), axis=0)
    total = p.sum(axis=0) + 1e-20
    band = (f >= 0.5) & (f <= 50)
    pb = p[band] / (p[band].sum(axis=0) + 1e-20)
    ent = -(pb * np.log(pb + 1e-20)).sum(axis=0) / np.log(band.sum())
    return {
        "spectral_entropy": float(np.mean(ent)),
        "dominant_hz": float(np.median(f[band][np.argmax(p[band], axis=0)])),
        "baseline_power_frac": float(np.mean(p[f < 0.5].sum(axis=0) / total)),
        "hf_power_frac": float(np.mean(p[f > 40].sum(axis=0) / total)),
    }


def permutation_entropy(x: np.ndarray, order: int = 4, delay: int = 2) -> float:
    from math import factorial

    n = len(x) - (order - 1) * delay
    emb = np.stack([x[i * delay : i * delay + n] for i in range(order)], axis=-1)
    ranks = np.argsort(emb, axis=-1)
    codes = (ranks * order ** np.arange(order)).sum(axis=-1)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(factorial(order)))


def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    r = r_frac * np.std(x)
    if r == 0:
        return float("nan")
    n = len(x) - m
    emb_m = np.stack([x[i : i + n] for i in range(m)], axis=1)
    emb_m1 = np.stack([x[i : i + n] for i in range(m + 1)], axis=1)
    iu = np.triu_indices(n, 1)
    dm = np.max(np.abs(emb_m[:, None, :] - emb_m[None, :, :]), axis=2)[iu]
    dm1 = np.max(np.abs(emb_m1[:, None, :] - emb_m1[None, :, :]), axis=2)[iu]
    b, a = np.sum(dm <= r), np.sum(dm1 <= r)
    return float(-np.log(a / b)) if a > 0 and b > 0 else float("nan")


# ---------------------------------------------------------------- per-record work
def forecasts(sig: np.ndarray, hist_len: int, horizon: int, peaks_full: np.ndarray | None):
    hist = sig[:hist_len]
    hist_peaks = rpeaks(hist[:, 1])
    period, acf_peak = estimate_period(hist, hist_peaks)
    out = {
        "zero": np.zeros((horizon, 12)),
        "mean": np.broadcast_to(hist.mean(axis=0), (horizon, 12)),
        "median": np.broadcast_to(np.median(hist, axis=0), (horizon, 12)),
        "last": np.broadcast_to(hist[-1], (horizon, 12)),
        "lastsec": np.broadcast_to(hist[-FS:].mean(axis=0), (horizon, 12)),
        "sn1": seasonal(hist, period, horizon, 1),
        "sn4": seasonal(hist, period, horizon, K_CYCLES),
    }
    if peaks_full is not None:
        out["oracle"] = oracle_timing(sig, peaks_full, hist_len, horizon, K_CYCLES)
    return out, hist_peaks, period, acf_peak


def process(task: tuple[int, int, int, str]) -> dict | None:
    import wfdb

    ecg_id, patient_id, fold, path = task
    sig = wfdb.rdsamp(path)[0].astype(np.float64)
    rec: dict = {"ecg_id": ecg_id, "patient_id": patient_id, "fold": fold}
    if sig.shape != (N_TOTAL, 12) or not np.isfinite(sig).all():
        rec["invalid"] = True
        return rec
    rec["invalid"] = False
    peaks_full = rpeaks(sig[:, 1])
    preds, hist_peaks, period, acf_peak = forecasts(sig, H, F, peaks_full)
    y = sig[H:]
    rr_fut = plausible_rr(peaks_full[peaks_full >= H - FS // 2])
    if len(rr_fut):
        # Diagnostic: correct future mean period, but no knowledge of beat-to-beat jitter.
        preds["sn4_trueperiod"] = seasonal(sig[:H], float(np.median(rr_fut)), F, K_CYCLES)

    abs_h = np.full((len(PREDICTORS), F), np.nan, dtype=np.float32)
    sq_h = np.full((len(PREDICTORS), F), np.nan, dtype=np.float32)
    for i, name in enumerate(PREDICTORS):
        p = preds.get(name)
        if p is None:
            continue
        err = y - p
        abs_h[i] = np.abs(err).mean(axis=1)
        sq_h[i] = (err**2).mean(axis=1)
        rec[f"mae_{name}"] = float(np.abs(err).mean())
        rec[f"rmse_{name}"] = float(np.sqrt((err**2).mean()))
        if name in DTW_PREDICTORS:
            rec[f"dtw_{name}"] = float(
                np.mean([dtw_band(y[:, j], p[:, j], DTW_BAND) for j in range(12)])
            )
    # Sufficient statistics for the analytic damped blend mean + a(h) * (sn4 - mean).
    resid = y - preds["mean"]
    d = preds["sn4"] - preds["mean"]
    blend = np.stack([(resid**2).mean(axis=1), (resid * d).mean(axis=1), (d**2).mean(axis=1)])

    # Rhythm descriptors.
    rr_hist = plausible_rr(hist_peaks)
    rr_full = plausible_rr(peaks_full)
    rec.update(
        period_s=period / FS,
        acf_peak=acf_peak,
        n_peaks_hist=len(hist_peaks),
        n_peaks_full=len(peaks_full),
        hr_bpm=60 * FS / np.median(rr_full) if len(rr_full) else np.nan,
        rr_cv_hist=np.std(rr_hist) / np.mean(rr_hist) if len(rr_hist) >= 2 else np.nan,
        rr_cv_full=np.std(rr_full) / np.mean(rr_full) if len(rr_full) >= 2 else np.nan,
        rr_shift=(
            abs(np.median(rr_fut) - np.median(rr_hist)) / np.median(rr_hist)
            if len(rr_fut) and len(rr_hist)
            else np.nan
        ),
        period_error=(
            abs(period - np.median(rr_fut)) / np.median(rr_fut) if len(rr_fut) else np.nan
        ),
        signal_std=float(sig[:H].std(axis=0).mean()),
        perm_entropy=float(np.mean([permutation_entropy(sig[:H, j]) for j in range(12)])),
        sample_entropy_II=sample_entropy(sig[:H, 1]),
    )
    rec.update(spectral_features(sig[:H]))

    # Alternative splits 8->2 and 9->1 (and 7->3 for symmetry): per lead-time MAE.
    split_h = np.full((len(SPLITS), 2, F), np.nan, dtype=np.float32)
    for s, h_len in enumerate(SPLITS):
        p, *_ = forecasts(sig, h_len, N_TOTAL - h_len, None)
        yy = sig[h_len:]
        split_h[s, 0, : N_TOTAL - h_len] = np.abs(yy - p["sn4"]).mean(axis=1)
        split_h[s, 1, : N_TOTAL - h_len] = np.abs(yy - p["mean"]).mean(axis=1)

    # Fixed target window 9-10 s, varying context length.
    target = sig[900:]
    for ctx in CONTEXTS:
        p, *_ = forecasts(sig[900 - ctx :], ctx, FS, None)
        rec[f"ctx{ctx}_sn4"] = float(np.abs(target - p["sn4"]).mean())
        rec[f"ctx{ctx}_mean"] = float(np.abs(target - p["mean"]).mean())

    rec["_abs_h"] = abs_h
    rec["_sq_h"] = sq_h
    rec["_blend"] = blend.astype(np.float32)
    rec["_split_h"] = split_h
    return rec


# ---------------------------------------------------------------- statistics
def cluster_boot_ratio(num, den, groups, rng, n_boot=N_BOOT):
    """Bootstrap ratio sum(num)/sum(den), resampling patient clusters."""
    df = pd.DataFrame({"n": num, "d": den, "g": groups}).groupby("g").sum()
    n_arr, d_arr = df["n"].to_numpy(), df["d"].to_numpy()
    idx = rng.integers(0, len(df), size=(n_boot, len(df)))
    stats = n_arr[idx].sum(axis=1) / d_arr[idx].sum(axis=1)
    return float(n_arr.sum() / d_arr.sum()), *np.percentile(stats, [2.5, 97.5])


def patient_macro(values, groups):
    return pd.Series(values).groupby(np.asarray(groups)).mean()


def cluster_boot_mean(values, groups, rng, n_boot=N_BOOT):
    """Patient-macro mean and percentile CI, resampling patients."""
    pm = patient_macro(values, groups).to_numpy()
    idx = rng.integers(0, len(pm), size=(n_boot, len(pm)))
    return float(pm.mean()), *np.percentile(pm[idx].mean(axis=1), [2.5, 97.5])


def fmt_ci(est, lo, hi, digits=4):
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ---------------------------------------------------------------- main
def load_metadata(root: Path) -> pd.DataFrame:
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db = db[db.strat_fold <= 9].copy()  # fold 10 sealed
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    codes = db.scp_codes.apply(ast.literal_eval)
    diag = scp[scp.diagnostic == 1].diagnostic_class.to_dict()
    for cls in SUPERCLASSES:
        db[f"sc_{cls}"] = codes.apply(lambda c, cls=cls: any(diag.get(k) == cls for k in c))
    for rhy in RHYTHM_GROUPS:
        db[f"rh_{rhy}"] = codes.apply(lambda c, rhy=rhy: rhy in c)
    for flag in QUALITY_FLAGS:
        db[f"q_{flag}"] = db[flag].notna()
    db["q_any"] = db[[f"q_{f}" for f in QUALITY_FLAGS]].any(axis=1)
    return db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("PTB-XL"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase0_feasibility"))
    ap.add_argument("--cache", type=Path, default=Path("results/phase0"))
    ap.add_argument("--workers", type=int, default=38)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(args.data)
    if args.limit:
        meta = meta.sample(args.limit, random_state=SEED)
    tasks = [
        (int(i), int(r.patient_id), int(r.strat_fold), str(args.data / r.filename_lr))
        for i, r in meta.iterrows()
    ]
    t0 = time.time()
    with Pool(args.workers) as pool:
        recs = [r for r in pool.imap(process, tasks, chunksize=16) if r is not None]
    runtime = time.time() - t0
    print(f"processed {len(recs)} records in {runtime:.0f}s", flush=True)

    invalid = [r["ecg_id"] for r in recs if r["invalid"]]
    recs = [r for r in recs if not r["invalid"]]
    abs_h = np.stack([r.pop("_abs_h") for r in recs])
    sq_h = np.stack([r.pop("_sq_h") for r in recs])
    blend = np.stack([r.pop("_blend") for r in recs])
    split_h = np.stack([r.pop("_split_h") for r in recs])
    df = pd.DataFrame(recs).set_index("ecg_id").join(meta.drop(columns=["patient_id"]))
    np.savez_compressed(
        args.cache / "arrays.npz",
        abs_h=abs_h,
        sq_h=sq_h,
        blend=blend,
        split_h=split_h,
        ecg_id=df.index.to_numpy(),
    )
    df.drop(columns=["report", "scp_codes"]).to_csv(args.cache / "per_record.csv")

    rng = np.random.default_rng(SEED)
    summary: dict = {
        "n_records": len(df),
        "n_patients": int(df.patient_id.nunique()),
        "invalid_records": invalid,
        "runtime_s": round(runtime, 1),
    }
    pid = df.patient_id.to_numpy()
    has_oracle = df.mae_oracle.notna().to_numpy()
    summary["oracle_coverage"] = float(has_oracle.mean())
    has_all = has_oracle & df.mae_sn4_trueperiod.notna().to_numpy()
    summary["all_forecasts_coverage"] = float(has_all.mean())

    # Headline table (records where every forecast incl. oracles is defined, so rows are paired).
    sub = df[has_all]
    rows = []
    for name in PREDICTORS:
        mae = cluster_boot_mean(sub[f"mae_{name}"], sub.patient_id, rng)
        rmse = cluster_boot_mean(sub[f"rmse_{name}"], sub.patient_id, rng)
        sk = cluster_boot_ratio(sub[f"mae_{name}"], sub.mae_mean, sub.patient_id, rng)
        skm = cluster_boot_ratio(sub[f"mae_{name}"], sub.mae_median, sub.patient_id, rng)
        row = {
            "forecast": name,
            "mae": fmt_ci(*mae),
            "rmse": fmt_ci(*rmse),
            "skill_vs_mean": fmt_ci(1 - sk[0], 1 - sk[2], 1 - sk[1], 3),
            "skill_vs_median": fmt_ci(1 - skm[0], 1 - skm[2], 1 - skm[1], 3),
            "dtw": "",
            "mae_point": mae[0],
            "rmse_point": rmse[0],
            "dtw_point": np.nan,
        }
        if name in DTW_PREDICTORS:
            dtw = cluster_boot_mean(sub[f"dtw_{name}"], sub.patient_id, rng)
            row["dtw"], row["dtw_point"] = fmt_ci(*dtw), dtw[0]
        rows.append(row)
    headline = pd.DataFrame(rows)
    headline.to_csv(args.out / "table_headline.csv", index=False)

    # Paired differences (H0.1, H0.3).
    def paired(a, b, frame):
        diff = (frame[a] - frame[b]).to_numpy()
        return cluster_boot_mean(diff, frame.patient_id, rng)

    summary["delta_sn4_minus_mean"] = paired("mae_sn4", "mae_mean", df)
    summary["delta_sn4_minus_last"] = paired("mae_sn4", "mae_last", df)
    summary["delta_sn4_minus_sn1"] = paired("mae_sn4", "mae_sn1", df)
    summary["delta_oracle_minus_sn4"] = paired("mae_oracle", "mae_sn4", sub)
    ratio = cluster_boot_ratio(sub.mae_oracle, sub.mae_sn4, sub.patient_id, rng)
    summary["oracle_relative_reduction"] = (1 - ratio[0], 1 - ratio[2], 1 - ratio[1])
    ratio = cluster_boot_ratio(sub.mae_sn4_trueperiod, sub.mae_sn4, sub.patient_id, rng)
    summary["trueperiod_relative_reduction"] = (1 - ratio[0], 1 - ratio[2], 1 - ratio[1])
    summary["dtw_delta_sn4_minus_median"] = paired("dtw_sn4", "dtw_median", df)
    summary["dtw_delta_sn4_minus_mean"] = paired("dtw_sn4", "dtw_mean", df)
    summary["dtw_delta_sn4_minus_zero"] = paired("dtw_sn4", "dtw_zero", df)

    # Horizon curves and skill horizon (H0.2).
    i_mean, i_sn4 = PREDICTORS.index("mean"), PREDICTORS.index("sn4")
    curves = pd.DataFrame(
        {name: np.nanmean(abs_h[:, i], axis=0) for i, name in enumerate(PREDICTORS)}
    )
    curves.index = np.arange(F) / FS
    curves.index.name = "lead_time_s"
    curves.to_csv(args.out / "table_horizon_mae.csv", float_format="%.5f")
    skill_curve = 1 - curves["sn4"] / curves["mean"]
    smooth = skill_curve.rolling(10, min_periods=1).mean()
    nonpos = np.where(smooth.to_numpy() <= 0)[0]
    summary["skill_horizon_s"] = float(nonpos[0] / FS) if len(nonpos) else None
    for lo_s, hi_s in ((0, 1), (1, 2), (2, 3)):
        sl = slice(lo_s * FS, hi_s * FS)
        for pred in ("sn1", "sn4", "sn4_trueperiod", "oracle"):
            ip = PREDICTORS.index(pred)
            o = ~np.isnan(abs_h[:, ip, 0])
            for ref in ("mean", "median"):
                ir = PREDICTORS.index(ref)
                res = cluster_boot_ratio(
                    abs_h[o, ip, sl].mean(1), abs_h[o, ir, sl].mean(1), pid[o], rng
                )
                summary[f"skill_{pred}_vs_{ref}_{lo_s}-{hi_s}s"] = (
                    1 - res[0],
                    1 - res[2],
                    1 - res[1],
                )
    skill_med = 1 - curves["sn4"] / curves["median"]
    nonpos = np.where(skill_med.rolling(10, min_periods=1).mean().to_numpy() <= 0)[0]
    summary["skill_horizon_vs_median_s"] = float(nonpos[0] / FS) if len(nonpos) else None
    summary["delta_sn4_minus_median"] = paired("mae_sn4", "mae_median", df)
    summary["delta_sn1_minus_median"] = paired("mae_sn1", "mae_median", df)
    summary["delta_sn1_minus_mean"] = paired("mae_sn1", "mae_mean", df)

    # Damped blend fitted on folds 1-8, evaluated on fold 9 (RMSE metric).
    train, val = df.fold.to_numpy() <= 8, df.fold.to_numpy() == 9
    e2, c, q = (blend[:, j] for j in range(3))
    alpha = c[train].sum(0) / q[train].sum(0)
    mse_blend = e2[val] - 2 * alpha * c[val] + alpha**2 * q[val]
    summary["blend_alpha_by_second"] = [
        float(alpha[s * FS : (s + 1) * FS].mean()) for s in range(3)
    ]
    summary["fold9_rmse"] = {
        "mean": float(np.sqrt(e2[val].mean())),
        "sn4": float(np.sqrt(sq_h[val, i_sn4].mean())),
        "damped_blend": float(np.sqrt(mse_blend.mean())),
    }
    pd.Series(alpha, index=np.arange(F) / FS, name="alpha").to_csv(
        args.out / "table_blend_alpha.csv"
    )

    # Fold 9 comparison with Week 5 reported numbers (H0.6).
    f9 = df[df.fold == 9]
    summary["fold9_pooled_mae"] = {n: float(f9[f"mae_{n}"].mean()) for n in PREDICTORS}
    summary["fold9_patient_macro_mae"] = {
        n: float(patient_macro(f9[f"mae_{n}"], f9.patient_id).mean()) for n in PREDICTORS
    }

    # Stratification (H0.4).
    group_rows = []
    groups = (
        [("rhythm", r, df[f"rh_{r}"]) for r in RHYTHM_GROUPS]
        + [("superclass", s, df[f"sc_{s}"]) for s in SUPERCLASSES]
        + [("quality", "flag: " + f, df[f"q_{f}"]) for f in QUALITY_FLAGS]
        + [("quality", "no quality flag", ~df.q_any), ("all", "all", df.q_any | ~df.q_any)]
    )
    for kind, name, mask in groups:
        g = df[mask.to_numpy()]
        if len(g) < 20:
            continue
        sk = cluster_boot_ratio(g.mae_sn4, g.mae_mean, g.patient_id, rng)
        go = g[g.mae_oracle.notna()]
        ok = cluster_boot_ratio(go.mae_oracle, go.mae_mean, go.patient_id, rng)
        s1 = cluster_boot_ratio(g.mae_sn1, g.mae_median, g.patient_id, rng)
        group_rows.append(
            {
                "kind": kind,
                "group": name,
                "n_records": len(g),
                "n_patients": g.patient_id.nunique(),
                "mae_mean": g.mae_mean.mean(),
                "mae_median": g.mae_median.mean(),
                "mae_sn1": g.mae_sn1.mean(),
                "mae_sn4": g.mae_sn4.mean(),
                "mae_oracle": go.mae_oracle.mean(),
                "skill_sn4": 1 - sk[0],
                "skill_sn4_lo": 1 - sk[2],
                "skill_sn4_hi": 1 - sk[1],
                "skill_sn1_vs_median": 1 - s1[0],
                "skill_sn1_vs_median_lo": 1 - s1[2],
                "skill_sn1_vs_median_hi": 1 - s1[1],
                "skill_oracle": 1 - ok[0],
                "skill_oracle_lo": 1 - ok[2],
                "skill_oracle_hi": 1 - ok[1],
                "rr_cv_median": g.rr_cv_hist.median(),
                "hr_median": g.hr_bpm.median(),
            }
        )
    pd.DataFrame(group_rows).to_csv(args.out / "table_groups.csv", index=False, float_format="%.4f")
    rh_skill = {}
    for r in RHYTHM_GROUPS:
        g = df[df[f"rh_{r}"]]
        if len(g) >= 20:
            rh_skill[r] = 1 - abs_h[df[f"rh_{r}"].to_numpy(), i_sn4].mean(0) / abs_h[
                df[f"rh_{r}"].to_numpy(), i_mean
            ].mean(0)
    pd.DataFrame(rh_skill, index=np.arange(F) / FS).to_csv(
        args.out / "table_skill_by_rhythm_horizon.csv", float_format="%.4f"
    )

    # Drivers of predictability: Spearman with per-record skill.
    df["skill_sn4_rec"] = 1 - df.mae_sn4 / df.mae_mean
    drivers = [
        "rr_cv_hist",
        "rr_shift",
        "period_error",
        "acf_peak",
        "hr_bpm",
        "spectral_entropy",
        "perm_entropy",
        "sample_entropy_II",
        "baseline_power_frac",
        "hf_power_frac",
        "signal_std",
        "age",
    ]
    corr = (
        df[drivers + ["skill_sn4_rec"]]
        .corr(method="spearman")["skill_sn4_rec"]
        .drop("skill_sn4_rec")
    )
    corr.to_csv(
        args.out / "table_drivers_spearman.csv", header=["spearman_rho"], float_format="%.3f"
    )

    # Characteristics distribution.
    desc = df[drivers + ["period_s", "n_peaks_full"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    desc.T.to_csv(args.out / "table_signal_characteristics.csv", float_format="%.4f")

    # Splits (H0.5a): matched lead-time 0-1 s and full-horizon.
    split_rows = []
    for s, h_len in enumerate(SPLITS):
        hor = N_TOTAL - h_len
        sn, mn = split_h[:, s, 0], split_h[:, s, 1]
        split_rows.append(
            {
                "split": f"{h_len // FS}->{hor // FS}",
                "mae_sn4_full_horizon": float(np.nanmean(sn[:, :hor])),
                "mae_mean_full_horizon": float(np.nanmean(mn[:, :hor])),
                "skill_full_horizon": float(1 - np.nanmean(sn[:, :hor]) / np.nanmean(mn[:, :hor])),
                "mae_sn4_lead_0-1s": float(np.nanmean(sn[:, :FS])),
                "skill_lead_0-1s": float(1 - np.nanmean(sn[:, :FS]) / np.nanmean(mn[:, :FS])),
            }
        )
    pd.DataFrame(split_rows).to_csv(args.out / "table_splits.csv", index=False, float_format="%.5f")
    np.savetxt(
        args.out / "table_split_curves.csv",
        np.column_stack([np.arange(F) / FS] + [np.nanmean(split_h[:, s, 0], 0) for s in range(3)]),
        delimiter=",",
        header="lead_time_s,sn4_7to3,sn4_8to2,sn4_9to1",
        comments="",
        fmt="%.5f",
    )

    # Context length (H0.5b).
    ctx_rows = [
        {
            "context_s": c / FS,
            "mae_sn4": df[f"ctx{c}_sn4"].mean(),
            "mae_mean": df[f"ctx{c}_mean"].mean(),
        }
        for c in CONTEXTS
    ]
    pd.DataFrame(ctx_rows).to_csv(args.out / "table_context.csv", index=False, float_format="%.5f")

    summary["peak_detection"] = {
        "median_peaks_full": float(df.n_peaks_full.median()),
        "frac_lt3_peaks": float((df.n_peaks_full < 3).mean()),
        "median_period_error": float(df.period_error.median()),
        "frac_period_error_gt10pct": float((df.period_error > 0.10).mean()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "invalid_records"}, indent=2, default=float
        )
    )


if __name__ == "__main__":
    main()
