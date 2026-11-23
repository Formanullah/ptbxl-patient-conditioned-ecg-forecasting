"""Phase 7: data-efficiency study (learning, performance and cost curves).

See reports/phase7_data_efficiency/PREREGISTRATION.md.

    python research/phase7_data_efficiency.py
"""

from __future__ import annotations

import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("reports/phase7_data_efficiency")
CACHE = Path(".cache/data/phase7")
P5 = Path(".cache/data/phase5")
H, F = 700, 290
FRACTIONS = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
SEEDS = (42, 123, 999)
WORKERS, THREADS = 9, 4


def pool_and_subset(meta, fraction: float, seed: int):
    """Training pool (folds 1-8 minus the fixed holdout) and the nested patient subset."""
    from ptbxl_forecast.splits import patient_holdout

    parts = patient_holdout(meta, 0.2, seed=0)  # the same holdout as Phase 5
    pool = parts["train"]
    patients = np.sort(pool.patient_id.unique())
    order = np.random.default_rng(seed).permutation(patients)
    chosen = set(order[: max(1, round(fraction * len(patients)))])
    sub = pool[pool.patient_id.isin(chosen)]  # keeps the pool's record order
    return meta.index.get_indexer(sub.index), meta.index.get_indexer(parts["val"].index), sub


def cnn_job(spec: dict) -> dict:
    import torch

    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.models.cnn import CNNForecaster
    from ptbxl_forecast.preprocessing import Pipeline
    from ptbxl_forecast.training import predict, train_forecaster

    f, seed = spec["fraction"], spec["seed"]
    path = CACHE / f"cnn_f{f:.2f}_s{seed}.npz"
    if path.exists():
        return np.load(path, allow_pickle=True)["info"].item()
    torch.set_num_threads(THREADS)
    cfg = load_config()
    meta, sig, _ = load_dataset(cfg, "train", cache_dir=".cache/data")
    _, va_sig, _ = load_dataset(cfg, "val", cache_dir=".cache/data")
    rows, rows_ho, sub = pool_and_subset(meta, f, seed)
    pipe = Pipeline().fit(sig[rows, :H].astype(np.float64), sub.index)
    model = CNNForecaster(horizon=F)
    info = train_forecaster(model, pipe, sig[rows], sig[rows_ho], H, F, seed=seed, threads=THREADS)
    pred = predict(model, pipe, va_sig[:, :H].astype(np.float64))
    best = info["log"][info["best_epoch"] - 1]
    info.update(fraction=f, seed=seed, n_records=len(rows), n_patients=int(sub.patient_id.nunique()),
                gap=best["holdout_l1"] - best["train_l1"])  # fmt: skip
    np.savez(path, pred=pred.astype(np.float32), info=np.array(info, dtype=object))
    return info


def ridge_run(f: float, seed: int, meta, sig, xva) -> dict:
    from ptbxl_forecast.models.ridge import RidgeForecaster

    path = CACHE / f"ridge_f{f:.2f}_s{seed}.npz"
    if path.exists():
        return np.load(path, allow_pickle=True)["info"].item()
    rows, _, sub = pool_and_subset(meta, f, seed)
    x = sig[rows, :H].astype(np.float64).reshape(len(rows), -1)
    y = sig[rows, H : H + F].astype(np.float64).reshape(len(rows), -1)
    t0 = time.perf_counter()
    model = RidgeForecaster(seed=seed).fit(x, y, sub.patient_id.to_numpy())
    pred = model.predict(xva.reshape(len(xva), -1)).reshape(len(xva), F, 12)
    info = {"fraction": f, "seed": seed, "n_records": len(rows), "lambda": model.lam,
            "seconds": time.perf_counter() - t0}  # fmt: skip
    np.savez(path, pred=pred.astype(np.float32), info=np.array(info, dtype=object))
    return info


_EVAL = {}


def _score(path: str) -> pd.DataFrame:
    from ptbxl_forecast import metrics as M

    x, y, peaks = _EVAL["x"], _EVAL["y"], _EVAL["peaks"]
    pred = (
        np.load(path, allow_pickle=True)["pred"].astype(np.float64)
        if not path.startswith("ENS:")
        else None
    )
    if pred is None:
        files = path[4:].split("|")
        pred = np.mean(
            [np.load(p, allow_pickle=True)["pred"].astype(np.float64) for p in files], axis=0
        )
    return M.evaluate(x, y, pred, true_peaks=peaks)


def main() -> None:
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.stats import cluster_bootstrap, compare

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    specs = [{"fraction": f, "seed": s} for f in FRACTIONS[::-1] for s in SEEDS]  # largest first
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("spawn")) as pool:
        cnn_info = list(pool.map(cnn_job, specs))
    print(f"CNN runs done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    cfg = load_config()
    meta, sig, _ = load_dataset(cfg, "train", cache_dir=".cache/data")
    va_meta, va_sig, _ = load_dataset(cfg, "val", cache_dir=".cache/data")
    xva, yva = va_sig[:, :H].astype(np.float64), va_sig[:, H : H + F].astype(np.float64)
    ridge_info = [ridge_run(f, s, meta, sig, xva) for f in FRACTIONS for s in SEEDS]
    print("ridge runs done", flush=True)

    # Bitwise reproducibility of the 100% CNN runs against Phase 5.
    repro = {s: bool(np.array_equal(np.load(CACHE / f"cnn_f1.00_s{s}.npz")["pred"], np.load(P5 / f"cnn_default_s{s}.npz")["pred"]))
             for s in SEEDS}  # fmt: skip

    # Score every run and every per-fraction CNN ensemble.
    pid = va_meta.patient_id.to_numpy()
    _EVAL.update(x=xva, y=yva, peaks=pickle.load((P5 / "val_true_peaks.pkl").open("rb")))
    jobs = [
        str(CACHE / f"{lr}_f{f:.2f}_s{s}.npz")
        for lr in ("cnn", "ridge")
        for f in FRACTIONS
        for s in SEEDS
    ]
    ens = [
        "ENS:" + "|".join(str(CACHE / f"cnn_f{f:.2f}_s{s}.npz") for s in SEEDS) for f in FRACTIONS
    ]
    with Pool(20) as pool:
        scored = dict(zip(jobs + ens, pool.map(_score, jobs + ens), strict=True))

    rows = []
    for info, lr in [(i, "cnn") for i in cnn_info] + [(i, "ridge") for i in ridge_info]:
        df = scored[str(CACHE / f"{lr}_f{info['fraction']:.2f}_s{info['seed']}.npz")]
        row = {"learner": lr, **{k: v for k, v in info.items() if k != "log"}}
        for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms"):
            row[m] = float(pd.Series(df[m].to_numpy()).groupby(pid).mean().mean())
        row["share_timing_le_50ms"] = float((df.beat_timing_ms <= 50).mean())
        rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "table_runs.csv", index=False, float_format="%.5f")

    p6 = pd.read_parquet(".cache/data/phase6_scores.parquet")
    dev, _, _ = load_dataset(cfg, "dev", cache_dir=".cache/data")
    p6["ecg_id"] = dev.index.to_numpy()[p6.row]
    flat = (
        p6[p6.probe == "median (flat)"]
        .set_index("ecg_id")
        .loc[va_meta.index]
        .reset_index(drop=True)
    )
    sn4 = (
        p6[p6.probe == "sn4 (repeat cycles)"]
        .set_index("ecg_id")
        .loc[va_meta.index]
        .reset_index(drop=True)
    )
    full = scored[ens[-1]]
    ens_rows = []
    for f, key in zip(FRACTIONS, ens, strict=True):
        df = scored[key]
        r = {
            "fraction": f,
            "n_records_mean": runs[(runs.learner == "cnn") & (runs.fraction == f)].n_records.mean(),
        }
        for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms"):
            r[f"ens_{m}"] = cluster_bootstrap(df[m], pid, n_boot=200, seed=0)[0]
        r["ens_share_timing_le_50ms"] = float((df.beat_timing_ms <= 50).mean())
        c_full = compare(df, full, pid)
        r["d_mae_vs_100"], r["d_mae_vs_100_lo"], r["d_mae_vs_100_hi"] = (
            c_full["mae"][k] for k in ("diff", "lo", "hi")
        )
        c_flat = compare(df, flat, pid)
        r["beats_flat"] = c_flat["beats_reference"]
        for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms"):
            r[f"d_{m}_vs_flat"], r[f"d_{m}_vs_flat_lo"], r[f"d_{m}_vs_flat_hi"] = (
                c_flat[m][k] for k in ("diff", "lo", "hi")
            )
        c_sn4 = compare(df, sn4, pid)
        r["beats_sn4"] = c_sn4["beats_reference"]
        r["timing_vs_sn4_diff"] = c_sn4["beat_timing_ms"]["diff"]
        ens_rows.append(r)
    ensembles = pd.DataFrame(ens_rows)
    ensembles["mae_rel_to_100"] = ensembles.ens_mae / ensembles.ens_mae.iloc[-1] - 1
    ensembles.to_csv(OUT / "table_ensembles.csv", index=False, float_format="%.5f")

    e = ensembles.set_index("fraction")
    g = runs[runs.learner == "cnn"].groupby("fraction")
    within = e[(e.mae_rel_to_100 <= 0.01) & (e.d_mae_vs_100_lo <= 0) & (e.d_mae_vs_100_hi >= 0)]
    doubling = [(a, b) for a, b in ((0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.80))]
    dim = [
        f"{a:.0%}->{b:.0%}"
        for a, b in doubling
        if 1 - e.loc[b, "ens_mae"] / e.loc[a, "ens_mae"] < 0.01
    ]
    S = {
        "bitwise_reproduction_of_phase5": repro,
        "minimum_useful_fraction": float(e.index[e.beats_flat].min()) if e.beats_flat.any() else None,
        "saturation_fraction": float(within.index.min()) if len(within) else None,
        "diminishing_returns_first_doubling": dim[0] if dim else None,
        "H7_1_mae60_rel_to_100": float(e.loc[0.60, "mae_rel_to_100"]),
        "H7_1_pass": bool(e.loc[0.60, "mae_rel_to_100"] <= 0.01),
        "H7_2_timing_improvement_20_to_100": float(1 - e.loc[1.0, "ens_beat_timing_ms"] / e.loc[0.20, "ens_beat_timing_ms"]),
        "H7_2_100pct_beats_sn4_on_timing": bool(e.loc[1.0, "timing_vs_sn4_diff"] < 0),
        "H7_3_gap_5pct": float(g.gap.mean().loc[0.05]),
        "H7_3_gap_100pct": float(g.gap.mean().loc[1.00]),
    }  # fmt: skip
    S["H7_2_pass"] = bool(
        S["H7_2_timing_improvement_20_to_100"] < 0.10 and not S["H7_2_100pct_beats_sn4_on_timing"]
    )
    S["H7_3_pass"] = bool(S["H7_3_gap_5pct"] >= 2 * S["H7_3_gap_100pct"])
    S["power_law_fit"] = power_law(e.n_records_mean.to_numpy(), e.ens_mae.to_numpy())
    (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=str) + "\n")
    print(json.dumps(S, indent=2, default=str))
    print(ensembles.round(5).to_string())
    figures(runs, ensembles, flat, sn4, pid)


def power_law(n: np.ndarray, mae: np.ndarray) -> dict:
    """Descriptive fit MAE(n) = a + b * n**-c on the ensemble curve, with extrapolations."""
    from scipy.optimize import curve_fit

    def f(x, a, b, c):
        return a + b * x**-c

    (a, b, c), _ = curve_fit(f, n, mae, p0=(0.09, 1.0, 0.5), maxfev=20000)
    resid = mae - f(n, a, b, c)
    return {"asymptote_a": float(a), "b": float(b), "exponent_c": float(c),
            "max_abs_residual": float(np.abs(resid).max()),
            "pred_17392_all_train_folds": float(f(17392, a, b, c)),
            "pred_2x_pool": float(f(2 * n.max(), a, b, c))}  # fmt: skip


def figures(runs, ensembles, flat, sn4, pid):
    from ptbxl_forecast.stats import patient_macro

    def curve(a, lr, col, scale=1.0, **kw):
        g = runs[runs.learner == lr].groupby("fraction")
        a.errorbar(g.n_records.mean(), g[col].mean() * scale, yerr=g[col].std() * scale,
                   marker="o", capsize=3, **kw)  # fmt: skip

    metrics = [("mae", "MAE (mV)"), ("dtw_mae", "DTW-MAE (mV)"), ("template_mae", "Template MAE (mV)"),
               ("beat_timing_ms", "Beat timing (ms)")]  # fmt: skip
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for a, (m, label) in zip(ax.ravel()[:4], metrics, strict=True):
        for lr, color in (("cnn", "#2171b5"), ("ridge", "#ff7f0e")):
            curve(a, lr, m, color=color, label=f"{lr} (mean ± SD over 3 seeds)")
        a.axhline(patient_macro(flat[m], pid), color="grey", ls="--", label="flat (history median)")
        a.axhline(patient_macro(sn4[m], pid), color="#2ca02c", ls=":", label="sn4 (repeat cycles)")
        a.set(
            xscale="log", xlabel="training records (log)", ylabel=label, title=f"{label} on fold 9"
        )
        a.legend(fontsize=7)
    curve(ax[1, 1], "cnn", "seconds", 1 / 60, label="CNN training (4 threads)")
    curve(ax[1, 1], "ridge", "seconds", 1 / 60, label="ridge fit (multithreaded BLAS)")
    ax[1, 1].set(xscale="log", xlabel="training records (log)", ylabel="wall-clock minutes",
                 title="Cost curve")  # fmt: skip
    ax[1, 1].legend(fontsize=8)
    curve(ax[1, 2], "cnn", "gap", color="#d62728")
    ax[1, 2].axhline(0, color="k", lw=0.8)
    ax[1, 2].set(xscale="log", xlabel="training records (log)", ylabel="holdout L1 - train L1 at best epoch",
                 title="Generalization gap (CNN)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig1_learning_curves.png", dpi=140)


if __name__ == "__main__":
    main()
