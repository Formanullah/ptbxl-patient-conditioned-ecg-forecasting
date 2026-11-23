"""Phase 3: data quality assessment for PTB-XL folds 1-9 (fold 10 sealed).

See reports/phase3_quality/PREREGISTRATION.md for definitions, hypotheses and the
exclusion rule. Writes tables, figures and summary.json.

    python research/phase3_quality.py --data PTB-XL --out reports/phase3_quality
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal as sps  # noqa: E402
from scipy import stats  # noqa: E402

from ptbxl_forecast.data import LEADS, load_metadata, load_signals  # noqa: E402

FS, N = 100, 1000
SEED = 20261122
_SIG = None
_SOS_BW = sps.butter(2, 0.67, "lowpass", fs=FS, output="sos")
_SOS_HP15 = sps.butter(4, 15, "highpass", fs=FS, output="sos")
FP_EDGES = np.linspace(0.3, 45, 65)  # 64 frequency bins for the shift-invariant fingerprint


def _init(path: str) -> None:
    global _SIG
    _SIG = np.load(path, mmap_mode="r")


def longest_run(mask: np.ndarray) -> int:
    """Longest run of True values."""
    if not mask.any():
        return 0
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.flatnonzero(np.diff(padded))
    return int((d[1::2] - d[::2]).max())


def peaks(x: np.ndarray, method: str) -> np.ndarray:
    import neurokit2 as nk

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.ecg_clean(
                x, sampling_rate=FS, method="neurokit" if method == "neurokit" else method
            )
            _, info = nk.ecg_peaks(clean, sampling_rate=FS, method=method)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def bsqi(a: np.ndarray, b: np.ndarray, tol: int = 15) -> float:
    if len(a) == 0 and len(b) == 0:
        return np.nan
    matched = sum(np.any(np.abs(b - p) <= tol) for p in a)
    return matched / (len(a) + len(b) - matched)


def tsqi(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per lead: mean correlation of each beat (-0.25 s..+0.45 s) with the median beat."""
    r = r[(r >= 25) & (r < N - 45)]
    if len(r) < 3:
        return np.full(x.shape[1], np.nan)
    beats = np.stack([x[p - 25 : p + 45] for p in r])  # (beats, 70, 12)
    template = np.median(beats, axis=0)
    b = beats - beats.mean(1, keepdims=True)
    t = template - template.mean(0, keepdims=True)
    num = (b * t).sum(1)
    den = np.sqrt((b**2).sum(1) * (t**2).sum(0)) + 1e-12
    return (num / den).mean(0)


def band(f, p, lo, hi):
    return p[(f >= lo) & (f < hi)].sum(0)


def analyse(i: int) -> dict:
    x = np.asarray(_SIG[i], dtype=np.float64)
    out: dict = {"row": i}
    d = np.diff(x, axis=0)
    out["dropout_run"] = max(longest_run(d[:, j] == 0) + 1 for j in range(12))
    plateau = 0
    for j in range(12):
        for v in (x[:, j].max(), x[:, j].min()):
            if abs(v) >= 1:
                plateau = max(plateau, longest_run(x[:, j] == v))
    out["plateau_run"] = plateau
    out["max_abs"] = float(np.abs(x).max())

    low = sps.sosfiltfilt(_SOS_BW, x, axis=0)
    bw_pp = np.ptp(low, axis=0)
    f, p = sps.welch(x, fs=FS, nperseg=256, axis=0)
    bas = 1 - band(f, p, 0, 1) / band(f, p, 0, 40)
    hf = band(f, p, 40, 50.1) / band(f, p, 0, 50.1)
    psq = band(f, p, 5, 15) / band(f, p, 5, 40)
    ksq = stats.kurtosis(x, axis=0, fisher=False)
    hp = sps.sosfiltfilt(_SOS_HP15, x, axis=0)
    win_rms = np.sqrt((hp.reshape(10, 100, 12) ** 2).mean(1))  # (10 windows, 12)
    burst = win_rms.max(0) / (np.median(win_rms, 0) + 1e-9)

    r_nk = peaks(x[:, 1], "neurokit")
    r_el = peaks(x[:, 1], "elgendi2010")
    t = tsqi(x, r_nk)
    for name, v in (("bw_pp", bw_pp), ("basSQI", bas), ("hf_ratio", hf), ("pSQI", psq),
                    ("kSQI", ksq), ("burst", burst), ("tSQI", t)):  # fmt: skip
        out[f"{name}_med"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        out[f"{name}_max"] = float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
        out[f"{name}_min"] = float(np.nanmin(v)) if np.isfinite(v).any() else np.nan
    out["bSQI"] = bsqi(r_nk, r_el)
    out["n_beats"] = len(r_nk)

    I, II = x[:, 0], x[:, 1]  # noqa: E741
    derived = {"III": II - I, "aVR": -(I + II) / 2, "aVL": I - II / 2, "aVF": II - I / 2}
    out["lead_resid_max"] = float(
        max(np.sqrt(((x[:, LEADS.index(k)] - v) ** 2).mean()) for k, v in derived.items())
    )
    out["hash"] = hashlib.sha256(np.round(x * 1000).astype(np.int16).tobytes()).hexdigest()
    out["_tpl"] = template_vector(x, r_nk)
    return out


def template_vector(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Unit vector of the 12-lead median beat around detected R peaks (shift-invariant).

    Zero vector when fewer than 3 beats are usable."""
    r = r[(r >= 25) & (r < len(x) - 45)]
    if len(r) < 3:
        return np.zeros(70 * 12, np.float32)
    t = np.median(np.stack([x[p - 25 : p + 45] for p in r]), axis=0).ravel()
    t = t - t.mean()
    return (t / (np.linalg.norm(t) + 1e-12)).astype(np.float32)


def max_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int = 500) -> tuple[float, int]:
    """Max over lags (|lag| <= max_lag) of the Pearson correlation on the overlapping
    segment, averaged over the 12 leads. Exact per-lag normalization (cumulative sums)."""
    n = len(a)
    a = a - a.mean(0)
    b = b - b.mean(0)
    m = 2 * n
    xc = np.fft.irfft(np.conj(np.fft.rfft(a, m, axis=0)) * np.fft.rfft(b, m, axis=0), m, axis=0)
    ca, ca2 = np.cumsum(a, 0), np.cumsum(a**2, 0)
    cb, cb2 = np.cumsum(b, 0), np.cumsum(b**2, 0)
    zero = np.zeros((1, a.shape[1]))
    ca, ca2, cb, cb2 = (np.vstack([zero, c]) for c in (ca, ca2, cb, cb2))  # prefix sums, len n+1
    lags = np.arange(-max_lag, max_lag + 1)
    k = (n - np.abs(lags))[:, None]
    s_a = np.where(lags >= 0, 0, -lags)  # overlap: a[s_a : s_a+k] against b[s_b : s_b+k]
    s_b = np.where(lags >= 0, lags, 0)
    e_a, e_b = s_a + k[:, 0], s_b + k[:, 0]
    sab = xc[np.where(lags >= 0, lags, m + lags)]
    sa, sa2 = ca[e_a] - ca[s_a], ca2[e_a] - ca2[s_a]
    sb, sb2 = cb[e_b] - cb[s_b], cb2[e_b] - cb2[s_b]
    cov = sab - sa * sb / k
    var = (sa2 - sa**2 / k) * (sb2 - sb**2 / k)
    r = np.where(var > 1e-12, cov / np.sqrt(np.maximum(var, 1e-12)), 0.0).mean(1)
    best = int(np.argmax(r))
    return float(r[best]), int(lags[best])


def _verify(pair: tuple[int, int]) -> tuple[int, int, float, int]:
    i, j = pair
    c, lag = max_xcorr(np.asarray(_SIG[i], np.float64), np.asarray(_SIG[j], np.float64))
    return i, j, c, lag


def auroc(score: pd.Series, label: pd.Series) -> float:
    s, y = score.to_numpy(), label.to_numpy().astype(bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return np.nan
    u = stats.mannwhitneyu(s[y], s[~y]).statistic
    return float(u / (y.sum() * (~y).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("PTB-XL"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase3_quality"))
    ap.add_argument("--cache", type=Path, default=Path(".cache/data"))
    ap.add_argument("--phase0", type=Path, default=Path("results/phase0/per_record.csv"))
    ap.add_argument("--workers", type=int, default=38)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    S: dict = {}

    meta = load_metadata(args.data, range(1, 10))
    sig = load_signals(args.data, meta, cache_dir=args.cache, workers=args.workers)
    if not isinstance(sig, np.memmap):  # first run: the cache was just written; reopen it
        sig = load_signals(args.data, meta, cache_dir=args.cache)
    with Pool(args.workers, initializer=_init, initargs=(sig.filename,)) as pool:
        rows = pool.map(analyse, range(len(meta)), chunksize=32)
    tpls = np.stack([r.pop("_tpl") for r in rows])
    q = pd.DataFrame(rows).set_index(meta.index).drop(columns="row")
    q = q.join(meta[["patient_id", "strat_fold", "device", "validated_by_human", "recording_dt",
                     "q_baseline_drift", "q_static_noise", "q_burst_noise", "q_electrodes_problems",
                     "q_any", "height", "weight", "age"]])  # fmt: skip
    q["device_family"] = q.device.str.split().str[0]

    # ------------------------------------------------------------------ flags
    flags = {
        "dropout (>=0.5 s constant)": q.dropout_run >= 50,
        "plateau at extreme (>=50 ms, |v|>=1 mV)": q.plateau_run >= 5,
        "implausible amplitude (>5 mV)": q.max_abs > 5,
        "lead relation violated (>0.01 mV)": q.lead_resid_max > 0.01,
        "baseline wander p-p > 1 mV (median lead)": q.bw_pp_med > 1,
        "basSQI < 0.95 (median lead)": q.basSQI_med < 0.95,
        "kSQI < 5 (median lead)": q.kSQI_med < 5,
        "pSQI outside 0.5-0.8 (median lead)": ~q.pSQI_med.between(0.5, 0.8),
        "bSQI < 0.8 (lead II)": q.bSQI < 0.8,
        "tSQI < 0.5 (median lead)": q.tSQI_med < 0.5,
        "burst ratio > 5 (median lead)": q.burst_med > 5,
    }
    defect = ["dropout (>=0.5 s constant)", "plateau at extreme (>=50 ms, |v|>=1 mV)",
              "implausible amplitude (>5 mV)", "lead relation violated (>0.01 mV)"]  # fmt: skip

    # ------------------------------------------------------------------ duplicates
    dup_groups = q.groupby("hash").filter(lambda g: len(g) > 1)
    S["exact_duplicate_records"] = int(len(dup_groups))
    pid = q.patient_id.to_numpy()
    row_of = {e: k for k, e in enumerate(meta.index)}

    # (1) every pair of records from the same patient
    within_pairs = [(row_of[a], row_of[b]) for _, g in q.groupby("patient_id")
                    for a, b in combinations(g.index, 2)]  # fmt: skip
    # (2) cross-patient candidates, zero lag: exact mean-over-leads Pearson for all pairs
    z = np.asarray(sig, np.float32)
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-6)
    z = z.reshape(len(z), -1) / (12 * N) ** 0.5  # dot product = mean-over-leads Pearson
    zero_lag = []
    for s in range(0, len(z), 1000):
        sim = z[s : s + 1000] @ z.T
        ii, jj = np.nonzero(sim >= 0.9)
        zero_lag += [(i + s, j) for i, j in zip(ii, jj, strict=True) if i + s < j]
    S["zero_lag_pairs_ge_0.9"] = len(zero_lag)
    # (3) cross-patient candidates, any shift: 5 nearest neighbours by median-beat template
    tpl_nn = []
    for s in range(0, len(tpls), 2000):
        sim = tpls[s : s + 2000] @ tpls.T
        sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -1
        nn = np.argpartition(-sim, 5, axis=1)[:, :5]
        tpl_nn += [(min(i + s, j), max(i + s, j)) for i, row in enumerate(nn) for j in row]
    cross_cand = sorted({p for p in zero_lag + tpl_nn if pid[p[0]] != pid[p[1]]})
    S["cross_patient_candidates_verified"] = len(cross_cand)

    with Pool(args.workers, initializer=_init, initargs=(sig.filename,)) as pool:
        verified = pool.map(_verify, within_pairs + cross_cand, chunksize=64)
    res = pd.DataFrame(verified, columns=["i", "j", "xcorr", "lag"])
    res["a"], res["b"] = meta.index[res.i], meta.index[res.j]
    res["same_patient"] = pid[res.i] == pid[res.j]
    res["same_fold"] = q.strat_fold.to_numpy()[res.i] == q.strat_fold.to_numpy()[res.j]
    dt = q.recording_dt.to_numpy()
    res["gap_days"] = np.abs((dt[res.i] - dt[res.j]) / np.timedelta64(1, "D"))
    within = res[res.same_patient].copy()
    cross = res[~res.same_patient].copy()
    within.drop(columns=["i", "j"]).to_csv(
        args.cache / "phase3_within_patient_pairs.csv", index=False
    )

    # Validation of the candidate generators on synthetic duplicates (300 random records):
    # U = unshifted copy + 0.01 mV noise; S = shifted by up to 2 s (tail mirrored) + noise.
    rng = np.random.default_rng(SEED)
    hit_zero, hit_tpl, xc_u, xc_s, xc_clean = [], [], [], [], []
    for i in rng.choice(len(meta), 300, replace=False):
        x = np.asarray(sig[i], np.float64)
        u = x + rng.normal(0, 0.01, x.shape)
        shift = int(rng.integers(1, 201))
        s_ = np.concatenate([x[shift:], x[:shift][::-1]]) + rng.normal(0, 0.01, x.shape)
        zu = (u - u.mean(0)) / (u.std(0) + 1e-6)
        hit_zero.append(float(zu.ravel() / (12 * N) ** 0.5 @ z[i]) >= 0.9)
        t = template_vector(s_, peaks(s_[:, 1], "neurokit"))
        sims = tpls @ t
        hit_tpl.append(i in np.argpartition(-sims, 5)[:5])
        xc_u.append(max_xcorr(x, u)[0])
        xc_s.append(max_xcorr(x, s_)[0])
        xc_clean.append(max_xcorr(x, np.concatenate([x[shift:], x[:shift][::-1]]))[0])
    S["candidate_generator_validation"] = {
        "zero_lag_recall_unshifted_copies": float(np.mean(hit_zero)),
        "template_top5_recall_shifted_copies": float(np.mean(hit_tpl)),
        "verifier_recall_at_0.99_shifted_noise_free": float(np.mean(np.array(xc_clean) >= 0.99)),
        "verifier_recall_at_0.99_unshifted_noise_0.01mV": float(np.mean(np.array(xc_u) >= 0.99)),
        "verifier_recall_at_0.99_shifted_noise_0.01mV": float(np.mean(np.array(xc_s) >= 0.99)),
        "first_attempt_spectral_fingerprint_recall": 0.0,
    }
    within_dup = within[within.xcorr >= 0.99]
    S["near_duplicates"] = {
        "within_patient_pairs_checked": len(within),
        "within_patient_pairs_xcorr_ge_0.99": len(within_dup),
        "within_patient_pairs_xcorr_ge_0.95": int((within.xcorr >= 0.95).sum()),
        "within_patient_ge_0.99_same_day": int((within_dup.gap_days < 1).sum()),
        "cross_patient_pairs_xcorr_ge_0.99": int((cross.xcorr >= 0.99).sum()),
        "cross_patient_pairs_xcorr_ge_0.95": int((cross.xcorr >= 0.95).sum()),
        "cross_patient_ge_0.99_cross_fold": int(((cross.xcorr >= 0.99) & ~cross.same_fold).sum()),
        "cross_patient_max_xcorr": float(cross.xcorr.max()) if len(cross) else None,
    }
    paced = meta.rhythm_codes.apply(lambda r: "PACE" in r).to_numpy()
    high = res[res.xcorr >= 0.95]
    S["near_duplicates"]["pairs_ge_0.95_both_paced_share"] = (
        float((paced[high.i] & paced[high.j]).mean()) if len(high) else None
    )
    cross.sort_values("xcorr", ascending=False).head(200).drop(columns=["i", "j"]).to_csv(
        out / "table_cross_patient_top_pairs.csv", index=False, float_format="%.4f"
    )
    exact_extra = {i for _, g in dup_groups.groupby("hash") for i in g.index[1:]}
    dup_ids = set(within_dup.b) | set(cross[cross.xcorr >= 0.99].b) | exact_extra
    flags["near-duplicate of another record (xcorr>=0.99)"] = q.index.isin(dup_ids)
    defect.append("near-duplicate of another record (xcorr>=0.99)")

    # ------------------------------------------------------------------ prevalence and impact
    if args.phase0.exists():
        p0r = pd.read_csv(args.phase0, index_col="ecg_id")[["mae_median", "mae_mean", "mae_oracle"]]
        q = q.join(p0r)
    prev = []
    val = q.strat_fold == 9
    for name, m in flags.items():
        m = pd.Series(np.asarray(m, bool), index=q.index)
        row = {"issue": name, "defect": name in defect, "records": int(m.sum()),
               "pct_records": 100 * m.mean(), "patients": int(q.patient_id[m].nunique()),
               "pct_folds1_8": 100 * m[~val].mean(), "pct_fold9": 100 * m[val].mean()}  # fmt: skip
        for dev in sorted(q.device_family.unique()):
            row[f"pct_{dev}"] = 100 * m[q.device_family == dev].mean()
        if "mae_oracle" in q and m.any():
            row["oracle_mae_flagged"] = q.mae_oracle[m].mean()
            row["oracle_mae_unflagged"] = q.mae_oracle[~m].mean()
            row["median_fc_mae_flagged"] = q.mae_median[m].mean()
            row["median_fc_mae_unflagged"] = q.mae_median[~m].mean()
        prev.append(row)
    prev = pd.DataFrame(prev)
    prev.to_csv(out / "table_prevalence.csv", index=False, float_format="%.3f")
    any_defect = np.any([np.asarray(flags[k], bool) for k in defect], axis=0)
    any_sqi = np.any([np.asarray(flags[k], bool) for k in flags if k not in defect], axis=0)
    S["records_with_any_defect"] = int(any_defect.sum())
    S["pct_records_with_any_defect"] = float(100 * any_defect.mean())
    S["pct_records_with_any_sqi_flag"] = float(100 * any_sqi.mean())
    # Pre-registered rule, applied per issue: exclude a defect only if it affects < 1% of records.
    excluded_issues = [k for k in defect if np.asarray(flags[k], bool).mean() < 0.01]
    excl = np.any([np.asarray(flags[k], bool) for k in excluded_issues], axis=0)
    S["exclusion_rule"] = {
        "issues_excluded": excluded_issues,
        "issues_kept": [k for k in defect if k not in excluded_issues],
        "n_excluded_records": int(excl.sum()),
        "excluded_ecg_ids": sorted(int(i) for i in q.index[excl]),
    }

    # ------------------------------------------------------------------ hypotheses
    human = q.validated_by_human.astype(bool)
    h31 = {}
    for idx, flag, thr in (("bw_pp_med", "q_baseline_drift", 0.75), ("hf_ratio_med", "q_static_noise", 0.70),
                           ("burst_med", "q_burst_noise", 0.70)):  # fmt: skip
        h31[f"{idx} -> {flag}"] = {
            "auroc_all": auroc(q[idx], q[flag]),
            "auroc_human_validated": auroc(q[idx][human], q[flag][human]),
            "threshold": thr,
        }
        h31[f"{idx} -> {flag}"]["pass"] = h31[f"{idx} -> {flag}"]["auroc_all"] >= thr
    # Best single index per flag, for context (post hoc).
    cont = [c for c in q.columns if c.endswith(("_med", "_max", "_min"))] + ["bSQI"]
    best = {}
    for flag in ("q_baseline_drift", "q_static_noise", "q_burst_noise"):
        scores = {c: auroc(q[c], q[flag]) for c in cont}
        scores = {c: max(v, 1 - v) for c, v in scores.items() if np.isfinite(v)}
        k = max(scores, key=scores.get)
        best[flag] = {"index": k, "auroc_either_direction": scores[k]}
    S["H3_1"] = h31
    S["H3_1_pass"] = all(v["pass"] for v in h31.values())
    S["posthoc_best_index_per_flag"] = best
    if "mae_oracle" in q:
        rho = stats.spearmanr(q.bw_pp_med, q.mae_oracle, nan_policy="omit").statistic
        S["H3_2_spearman_bw_pp_vs_oracle_mae"] = float(rho)
        S["H3_2_pass"] = bool(rho >= 0.3)
        drivers = {c: float(stats.spearmanr(q[c], q.mae_oracle, nan_policy="omit").statistic)
                   for c in cont}  # fmt: skip
        S["posthoc_spearman_with_oracle_mae_top"] = dict(
            sorted(drivers.items(), key=lambda kv: -abs(kv[1]))[:8]
        )
    S["H3_3_pass"] = (
        S["exact_duplicate_records"] == 0
        and S["near_duplicates"]["cross_patient_pairs_xcorr_ge_0.99"] == 0
    )

    # Metadata plausibility (descriptive)
    S["metadata_issues"] = {
        "height_lt_100cm_adult": int(((q.height < 100) & (q.age >= 18)).sum()),
        "height_gt_220cm": int((q.height > 220).sum()),
        "weight_lt_20kg_adult": int(((q.weight < 20) & (q.age >= 18)).sum()),
        "weight_gt_180kg": int((q.weight > 180).sum()),
        "missing_share": {
            c: float(meta[c].isna().mean())
            for c in (
                "age",
                "sex",
                "height",
                "weight",
                "nurse",
                "site",
                "device",
                "heart_axis",
                "infarction_stadium1",
                "validated_by",
            )
        },  # fmt: skip
    }
    sqi_cols = ["dropout_run", "plateau_run", "max_abs", "bw_pp_med", "basSQI_med", "hf_ratio_med",
                "pSQI_med", "kSQI_med", "bSQI", "tSQI_med", "burst_med", "lead_resid_max"]  # fmt: skip
    q[sqi_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.to_csv(
        out / "table_index_distributions.csv", float_format="%.4f"
    )
    q.drop(columns=["hash"]).to_csv(args.cache / "phase3_per_record.csv")
    pd.Series(S["exclusion_rule"]["excluded_ecg_ids"], name="ecg_id").to_csv(
        out / "excluded_ecg_ids.csv", index=False
    )
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str) + "\n")
    figures(q, flags, within, sig, meta, h31, out)
    print(json.dumps(S, indent=2, default=str))


def figures(q, flags, within, sig, meta, h31, out):
    # Fig 1: index distributions with thresholds
    spec = [("bw_pp_med", "baseline wander p-p (mV)", 1, True), ("basSQI_med", "basSQI", 0.95, False),
            ("kSQI_med", "kSQI", 5, True), ("pSQI_med", "pSQI", (0.5, 0.8), False),
            ("bSQI", "bSQI (lead II)", 0.8, False), ("tSQI_med", "tSQI", 0.5, False),
            ("burst_med", "burst ratio", 5, True), ("hf_ratio_med", "HF ratio (40-50 Hz)", None, True)]  # fmt: skip
    fig, ax = plt.subplots(2, 4, figsize=(16, 6.5))
    for a, (col, label, thr, logx) in zip(ax.ravel(), spec, strict=True):
        v = q[col].dropna()
        bins = np.logspace(np.log10(max(v.min(), 1e-5)), np.log10(v.max()), 60) if logx else 60
        a.hist(v, bins=bins, color="#4c72b0")
        if logx:
            a.set_xscale("log")
        for t in thr if isinstance(thr, tuple) else (thr,):
            if t is not None:
                a.axvline(t, color="r", ls="--", lw=1)
        a.set(xlabel=label, ylabel="records")
    fig.suptitle("Signal-quality indices (folds 1-9; red: screening threshold)")
    fig.tight_layout()
    fig.savefig(out / "fig1_index_distributions.png", dpi=140)

    # Fig 2: ROC for H3.1
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (key, res) in zip(ax, h31.items(), strict=True):
        idx, flag = key.split(" -> ")
        s, y = q[idx].to_numpy(), q[flag].to_numpy().astype(bool)
        ok = np.isfinite(s)
        s, y = s[ok], y[ok]
        order = np.argsort(-s)
        tpr = np.cumsum(y[order]) / y.sum()
        fpr = np.cumsum(~y[order]) / (~y).sum()
        a.plot(fpr, tpr, color="#4c72b0")
        a.plot([0, 1], [0, 1], "k--", lw=0.8)
        a.set(xlabel="false-positive rate", ylabel="true-positive rate",
              title=f"{idx} vs {flag}\nAUROC {res['auroc_all']:.3f} (need {res['threshold']})")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "fig2_flag_agreement.png", dpi=140)

    # Fig 3: quality vs forecast difficulty
    if "mae_oracle" in q:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        d = q[["bw_pp_med", "mae_oracle"]].dropna()
        ax[0].hexbin(
            np.log10(d.bw_pp_med.clip(1e-3)),
            d.mae_oracle.clip(upper=0.4),
            gridsize=45,
            bins="log",
            cmap="Blues",
        )
        ax[0].set(xlabel="log10 baseline wander p-p (mV)", ylabel="Phase 0 oracle-timing MAE (mV)",
                  title="Baseline wander vs forecast error")  # fmt: skip
        d = q[["tSQI_med", "mae_oracle"]].dropna()
        ax[1].hexbin(
            d.tSQI_med, d.mae_oracle.clip(upper=0.4), gridsize=45, bins="log", cmap="Blues"
        )
        ax[1].set(
            xlabel="tSQI (beat consistency)",
            ylabel="oracle-timing MAE (mV)",
            title="Beat consistency vs error",
        )
        groups = {
            k: np.asarray(v, bool) for k, v in flags.items() if np.asarray(v, bool).sum() >= 20
        }
        names = list(groups)
        ratio = [q.mae_oracle[groups[k]].mean() / q.mae_oracle[~groups[k]].mean() for k in names]
        ax[2].barh(names[::-1], ratio[::-1], color="#c44e52")
        ax[2].axvline(1, color="k", lw=0.8)
        ax[2].set(xlabel="oracle MAE flagged / unflagged", title="Error ratio by issue (n >= 20)")
        ax[2].tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig3_quality_vs_difficulty.png", dpi=140)

    # Fig 4: within-patient similarity
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, m in (("same day", within.gap_days < 1), ("later", within.gap_days >= 1)):
        ax.hist(
            within.xcorr[m],
            bins=np.linspace(-0.2, 1, 61),
            alpha=0.6,
            label=f"{lab} (n={int(m.sum())})",
        )
    ax.axvline(0.99, color="r", ls="--")
    ax.set(xlabel="max 12-lead cross-correlation", ylabel="record pairs",
           title="Similarity of a patient's recordings")  # fmt: skip
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig4_within_patient_similarity.png", dpi=140)

    # Fig 5: one example per defect type (worst by its own index)
    t = np.arange(N) / FS
    examples = [("dropout", q.dropout_run.idxmax(), None), ("lead relation violated", q.lead_resid_max.idxmax(), None),
                ("baseline wander", q.bw_pp_med.idxmax(), None), ("burst noise", q.burst_med.idxmax(), None)]  # fmt: skip
    fig, ax = plt.subplots(len(examples), 1, figsize=(13, 2.6 * len(examples)))
    for a, (label, eid, _) in zip(ax, examples, strict=True):
        x = np.asarray(sig[meta.index.get_loc(eid)])
        if label == "lead relation violated":
            a.plot(t, x[:, 2], lw=0.8, label="III (stored)")
            a.plot(t, x[:, 1] - x[:, 0], lw=0.8, label="II - I")
            a.legend(fontsize=8)
        else:
            for j in range(12):
                a.plot(t, x[:, j], lw=0.6)
        a.set(
            title=f"{label}: ecg_id {eid} (fold {q.strat_fold[eid]}, {q.device_family[eid]})",
            ylabel="mV",
        )
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "fig5_defect_examples.png", dpi=120)


if __name__ == "__main__":
    main()
