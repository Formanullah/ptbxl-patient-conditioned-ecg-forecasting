"""Phase 6: evaluation-framework study. See reports/phase6_evaluation/PREREGISTRATION.md.

Scores probe forecasts (controlled perturbations of the truth) and realistic baselines
with every candidate metric, checks the pre-registered axioms, measures redundancy and
discriminative power, and writes the baseline reference table for later phases.

    python research/phase6_metrics.py
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal as sps  # noqa: E402

from ptbxl_forecast import metrics as M  # noqa: E402
from ptbxl_forecast.config import load_config  # noqa: E402
from ptbxl_forecast.data import load_dataset  # noqa: E402
from ptbxl_forecast.stats import cluster_bootstrap, paired_effect  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import phase0_feasibility as p0  # noqa: E402

OUT = Path("reports/phase6_evaluation")
H, F = 700, 290
SEED = 20261122
HIGHER_BETTER = {"pearson", "snr_db", "beat_f1"}
POST_HOC = {"beat_timing_ms"}
FAMILY = {
    "mae": "pointwise",
    "rmse": "pointwise",
    "mse": "pointwise",
    "mae_8lead": "pointwise",
    "snr_db": "pointwise",
    "pearson": "correlation",
    "dtw_mae": "timing-tolerant",
    "template_mae": "morphology",
    "r_amp_err": "morphology",
    "st_err": "morphology",
    "t_amp_err": "morphology",
    "beat_f1": "rhythm",
    "beat_timing_ms": "rhythm",
    "rr_err_ms": "rhythm",
}
# Pre-registered applicability: A1 for plausibility-oriented families, A3 for all but correlation,
# A4 for morphology, A5 for rhythm; A2 and A6 for all. A6 passes if NaN rate <= 1% (documented).
FAMILY_AXIOMS = {
    "pointwise": ["A2_monotone_shift", "A3_monotone_amplitude"],
    "correlation": ["A1_shift50_beats_flat", "A2_monotone_shift"],
    "timing-tolerant": ["A1_shift50_beats_flat", "A2_monotone_shift", "A3_monotone_amplitude"],
    "morphology": ["A1_shift50_beats_flat", "A2_monotone_shift", "A3_monotone_amplitude", "A4_penalizes_blur"],
    "rhythm": ["A1_shift50_beats_flat", "A2_monotone_shift", "A3_monotone_amplitude", "A5_penalizes_missing_beat"],
}  # fmt: skip
METRICS = ["mae", "rmse", "mse", "mae_8lead", "snr_db", "pearson", "dtw_mae", "template_mae",
           "r_amp_err", "st_err", "t_amp_err", "beat_f1", "beat_timing_ms", "rr_err_ms"]  # fmt: skip
PREFERENCE = ["mae", "rmse", "dtw_mae", "template_mae", "r_amp_err", "st_err", "t_amp_err",
              "beat_f1", "beat_timing_ms", "rr_err_ms", "pearson", "snr_db", "mse"]  # fmt: skip
SIG = None


def _init(arr):
    global SIG
    SIG = arr


def truth_peaks(i: int) -> np.ndarray:
    return M.detect_beats(SIG[i, :H], SIG[i, H : H + F])


def baselines(i: int) -> dict[str, np.ndarray]:
    x = SIG[i].astype(np.float64)
    preds, *_ = p0.forecasts(x, H, F, p0.rpeaks(x[:, 1]))
    return {"median (flat)": preds["median"], "sn4 (repeat cycles)": preds["sn4"],
            "oracle timing": preds["oracle"] if preds["oracle"] is not None else preds["sn4"]}  # fmt: skip


def score(args) -> pd.DataFrame:
    name, rows, yhat, peaks = args
    x, y = SIG[rows, :H], SIG[rows, H : H + F]
    df = M.evaluate(x, y, yhat, true_peaks=peaks)
    df.insert(0, "row", rows)
    df.insert(0, "probe", name)
    return df


def remove_beat(y: np.ndarray, peaks: np.ndarray) -> np.ndarray | None:
    full = peaks[(peaks >= M.PRE) & (peaks + M.POST <= len(y))]
    if len(full) == 0:
        return None
    p = full[0]
    out = y.copy()
    out[p - M.PRE : p + M.POST] = y[p + M.PR]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    meta, sig, _ = load_dataset(cfg, "dev", cache_dir=".cache/data")
    sig = np.ascontiguousarray(sig, dtype=np.float32)
    n = len(meta)
    pid = meta.patient_id.to_numpy()
    _init(sig)  # workers inherit SIG via fork (Linux); the array is not pickled
    with Pool(38) as pool:
        tpk = pool.map(truth_peaks, range(n), chunksize=64)
        base = pool.map(baselines, range(n), chunksize=64)

    x, y = sig[:, :H].astype(np.float64), sig[:, H : H + F].astype(np.float64)
    centre = x.mean(1, keepdims=True)
    rng = np.random.default_rng(SEED)
    lp = {c: sps.sosfiltfilt(sps.butter(4, c, "lowpass", fs=100, output="sos"), sig, axis=1)[:, H:].astype(np.float64)
          for c in (15, 5)}  # fmt: skip
    probes = {"perfect": y.copy()}
    for s in (2, 5, 10, 20):
        probes[f"late {10 * s} ms"] = sig[:, H - s : H + F - s].astype(np.float64)
    probes["amplitude x0.8"] = centre + 0.8 * (y - centre)
    probes["amplitude x0.5"] = centre + 0.5 * (y - centre)
    probes["low-pass 15 Hz"], probes["low-pass 5 Hz"] = lp[15], lp[5]
    probes["offset +0.1 mV"] = y + 0.1
    probes["noise 0.05 mV"] = y + rng.normal(0, 0.05, y.shape)
    rb = [remove_beat(y[i], tpk[i]) for i in range(n)]
    rb_rows = np.array([i for i in range(n) if rb[i] is not None])
    for name in ("median (flat)", "sn4 (repeat cycles)", "oracle timing"):
        probes[name] = np.stack([b[name] for b in base])

    jobs = []
    for name, yh in probes.items():
        for s in range(0, n, 500):
            rows = np.arange(s, min(s + 500, n))
            jobs.append((name, rows, yh[rows].astype(np.float32), [tpk[r] for r in rows]))
    for s in range(0, len(rb_rows), 500):
        rows = rb_rows[s : s + 500]
        jobs.append(
            ("one beat removed", rows, np.stack([rb[r] for r in rows]), [tpk[r] for r in rows])
        )
    with Pool(38) as pool:
        res = pd.concat(pool.map(score, jobs, chunksize=1), ignore_index=True)
    res["patient_id"] = pid[res.row]
    res["fold"] = meta.strat_fold.to_numpy()[res.row]
    res.to_parquet(".cache/data/phase6_scores.parquet") if _has_parquet() else res.to_csv(
        ".cache/data/phase6_scores.csv"
    )

    # ------------------------------------------------------------ patient-macro matrix
    mat = res.groupby("probe")[METRICS].apply(
        lambda g: pd.Series(
            {m: g.groupby(res.loc[g.index, "patient_id"])[m].mean().mean() for m in METRICS}
        )
    )
    nan_rate = res.groupby("probe")[METRICS].apply(lambda g: g.isna().mean())
    order = ["perfect", "late 20 ms", "late 50 ms", "late 100 ms", "late 200 ms", "amplitude x0.8",
             "amplitude x0.5", "low-pass 15 Hz", "low-pass 5 Hz", "offset +0.1 mV", "noise 0.05 mV",
             "one beat removed", "median (flat)", "sn4 (repeat cycles)", "oracle timing"]  # fmt: skip
    mat, nan_rate = mat.reindex(order), nan_rate.reindex(order)
    mat.to_csv(OUT / "table_probe_matrix.csv", float_format="%.5f")
    nan_rate.to_csv(OUT / "table_probe_nan_rate.csv", float_format="%.4f")

    def better(m, a, b):  # is probe a strictly better than probe b on metric m?
        va, vb = mat.loc[a, m], mat.loc[b, m]
        return bool(va > vb) if m in HIGHER_BETTER else bool(va < vb)

    ax = {}
    for m in METRICS:
        seq = ["perfect", "late 20 ms", "late 50 ms", "late 100 ms", "late 200 ms"]
        ax[m] = {
            "A1_shift50_beats_flat": better(m, "late 50 ms", "median (flat)"),
            "A1_shift20_beats_flat": better(m, "late 20 ms", "median (flat)"),
            "A2_monotone_shift": all(better(m, a, b) for a, b in zip(seq, seq[1:], strict=False)),
            "A3_monotone_amplitude": better(m, "perfect", "amplitude x0.8") and better(m, "amplitude x0.8", "amplitude x0.5"),
            "A4_penalizes_blur": better(m, "perfect", "low-pass 15 Hz") and better(m, "low-pass 15 Hz", "low-pass 5 Hz"),
            "A5_penalizes_missing_beat": better(m, "perfect", "one beat removed"),
            "A6_max_nan_rate": float(nan_rate[m].max()),
            "flat_beats_sn4": better(m, "median (flat)", "sn4 (repeat cycles)"),
        }  # fmt: skip
    axioms = pd.DataFrame(ax).T

    # ------------------------------------------------------------ redundancy on realistic forecasts
    real = res[res.probe.isin(["median (flat)", "sn4 (repeat cycles)", "oracle timing"])]
    rho = real[METRICS].corr(method="spearman")
    rho.to_csv(OUT / "table_redundancy_spearman.csv", float_format="%.3f")
    axioms["family"] = [FAMILY[m] for m in axioms.index]
    axioms["passes_family_axioms"] = [
        bool(
            all(axioms.loc[m, a] for a in FAMILY_AXIOMS[FAMILY[m]])
            and axioms.loc[m, "A6_max_nan_rate"] <= 0.01
        )
        for m in axioms.index
    ]
    axioms.to_csv(OUT / "table_axioms.csv")
    suite, dropped = [], {}
    for m in PREFERENCE:
        if not axioms.loc[m, "passes_family_axioms"]:
            dropped[m] = "fails family axioms"
            continue
        clash = [s for s in suite if abs(rho.loc[m, s]) > 0.9]
        if clash:
            dropped[m] = f"redundant with {clash[0]} (|rho|={abs(rho.loc[m, clash[0]]):.2f})"
            continue
        suite.append(m)

    # ------------------------------------------------------------ discriminative power
    disc = {}
    for m in METRICS:
        sign = -1 if m in HIGHER_BETTER else 1
        a = res[res.probe == "sn4 (repeat cycles)"].set_index("row")[m]
        b = res[res.probe == "median (flat)"].set_index("row")[m]
        c = res[res.probe == "oracle timing"].set_index("row")[m]
        disc[m] = {"sn4_vs_median": sign * paired_effect(a, b, pid[a.index]),
                   "oracle_vs_sn4": sign * paired_effect(c, a, pid[a.index])}  # fmt: skip
    disc = pd.DataFrame(disc).T
    disc.to_csv(OUT / "table_discrimination.csv", float_format="%.3f")

    # ------------------------------------------------------------ baseline reference table (fold 9 and dev)
    ref = []
    for scope, mask in (("fold 9 (validation)", res.fold == 9), ("folds 1-9", res.fold > 0)):
        for probe in ("median (flat)", "sn4 (repeat cycles)", "oracle timing"):
            sub = res[mask & (res.probe == probe)]
            row = {"scope": scope, "forecast": probe, "records": len(sub)}
            for m in METRICS:
                est, lo, hi = cluster_bootstrap(sub[m], sub.patient_id, n_boot=1000, seed=SEED)
                row[m], row[f"{m}_lo"], row[f"{m}_hi"] = est, lo, hi
            ref.append(row)
    pd.DataFrame(ref).to_csv(OUT / "table_baseline_reference.csv", index=False, float_format="%.5f")

    S = {
        "records": n,
        "one_beat_removed_records": int(len(rb_rows)),
        "true_beats_per_window_median": float(np.median([len(p) for p in tpk])),
        "windows_without_full_true_beat": float(
            np.mean([M.template(y[i], tpk[i]) is None for i in range(n)])
        ),
        "axioms": axioms.to_dict(orient="index"),
        "suite": suite,
        "dropped": dropped,
        "discrimination": disc.to_dict(orient="index"),
    }
    (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in S.items() if k != "axioms"}, indent=2, default=str))
    print(axioms.to_string())
    figures(mat, rho, sig, probes, tpk)


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def figures(mat, rho, sig, probes, tpk):
    norm = mat.copy()
    for m in norm:
        v = norm[m]
        lo, hi = v.min(), v.max()
        norm[m] = (v - lo) / (hi - lo) if hi > lo else 0
        if m in HIGHER_BETTER:
            norm[m] = 1 - norm[m]
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    im = ax[0].imshow(norm.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    ax[0].set_xticks(range(len(norm.columns)), norm.columns, rotation=60, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(norm)), norm.index, fontsize=8)
    for i in range(len(norm)):
        for j in range(len(norm.columns)):
            ax[0].text(j, i, f"{mat.iloc[i, j]:.3g}", ha="center", va="center", fontsize=6)
    ax[0].set_title("Metric value per probe (colour: 0 best -> 1 worst within each metric)")
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    im = ax[1].imshow(rho.abs().to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax[1].set_xticks(range(len(rho)), rho.columns, rotation=60, ha="right", fontsize=8)
    ax[1].set_yticks(range(len(rho)), rho.index, fontsize=8)
    for i in range(len(rho)):
        for j in range(len(rho)):
            ax[1].text(
                j, i, f"{rho.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w"
            )
    ax[1].set_title("|Spearman| between metrics across records (realistic forecasts)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_probe_matrix_and_redundancy.png", dpi=140)

    i = int(np.argmax([len(p) for p in tpk]))
    t = np.arange(990) / 100
    fig, axs = plt.subplots(2, 3, figsize=(16, 6), sharey=True)
    show = [
        "late 50 ms",
        "amplitude x0.5",
        "low-pass 5 Hz",
        "one beat removed",
        "median (flat)",
        "sn4 (repeat cycles)",
    ]
    for a, name in zip(axs.ravel(), show, strict=True):
        a.plot(t[600:], sig[i, 600:, 1], color="k", lw=1, label="truth (lead II)")
        if name == "one beat removed":
            yh = remove_beat(sig[i, H:, 1:2].astype(np.float64), tpk[i])
            yh = yh[:, 0] if yh is not None else sig[i, H:, 1]
        else:
            yh = probes[name][i, :, 1]
        a.plot(t[H:], yh, color="#d62728", lw=1, label=name)
        a.axvline(7, color="grey", ls="--", lw=0.8)
        a.set_title(name)
        a.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_probe_examples.png", dpi=130)


if __name__ == "__main__":
    main()
