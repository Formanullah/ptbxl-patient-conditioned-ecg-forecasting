"""Phase 4: data leakage analysis (H4.1-H4.4). See reports/phase4_leakage/PREREGISTRATION.md.

python research/phase4_leakage.py
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

from ptbxl_forecast.config import load_config  # noqa: E402
from ptbxl_forecast.data import SUPERCLASSES, load_dataset, load_metadata  # noqa: E402
from ptbxl_forecast.leakage import causality_violation  # noqa: E402
from ptbxl_forecast.splits import fold_assignments, group_kfold, official_split, patient_holdout  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import phase0_feasibility as p0  # noqa: E402

OUT = Path("reports/phase4_leakage")
SEEDS = (42, 123, 999)
H, F = 700, 290
SEG, CONT, STEP = 200, 490, 2  # 2 s matched segment (every 2nd sample), continuation length
LIB_LO, LIB_HI, Q_LO = SEG, 990 - CONT, 500  # library anchors in [200, 500]; query anchor >= 500


def peaks_pair(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R peaks (lead II) detected on the history only, and on the whole record."""
    return p0.rpeaks(x[:H].astype(np.float64)), p0.rpeaks(x.astype(np.float64))


def segment(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    seg = x[p - SEG : p]
    mean = seg.mean(0)
    return (seg[::STEP] - mean).ravel(), mean


# ------------------------------------------------------------------ memorization probe
class Probe:
    def __init__(self, meta: pd.DataFrame, sig: np.ndarray, peaks):
        self.meta, self.sig = meta, sig
        self.pid = meta.patient_id.to_numpy()
        lib_rec, lib_anchor, lib_feat, lib_mean = [], [], [], []
        q_anchor = np.full(len(meta), -1)
        q_feat = np.zeros((len(meta), SEG // STEP * 12), np.float32)
        q_mean = np.zeros((len(meta), 12), np.float32)
        for r, (hist_p, full_p) in enumerate(peaks):
            for p in full_p[(full_p >= LIB_LO) & (full_p <= LIB_HI)]:
                f, m = segment(sig[r], p)
                lib_rec.append(r), lib_anchor.append(p), lib_feat.append(f), lib_mean.append(m)
            cand = hist_p[(hist_p >= Q_LO) & (hist_p < H)]
            if len(cand):
                q_anchor[r] = cand[-1]
                q_feat[r], q_mean[r] = segment(sig[r], cand[-1])
        self.lib_rec = np.array(lib_rec)
        self.lib_anchor = np.array(lib_anchor)
        self.lib_feat = np.stack(lib_feat).astype(np.float32)
        self.lib_mean = np.stack(lib_mean).astype(np.float32)
        self.lib_sq = (self.lib_feat**2).sum(1)
        self.q_anchor, self.q_feat, self.q_mean = q_anchor, q_feat, q_mean

    def evaluate(self, train_rows: np.ndarray, val_rows: np.ndarray, k_list=(1, 5)) -> dict:
        lib = np.isin(self.lib_rec, train_rows)
        lf, lsq = self.lib_feat[lib], self.lib_sq[lib]
        lrec, lanc, lmean = self.lib_rec[lib], self.lib_anchor[lib], self.lib_mean[lib]
        val_rows = val_rows[self.q_anchor[val_rows] >= 0]
        kmax = max(k_list)
        err = {k: np.zeros(len(val_rows)) for k in k_list}
        same = np.zeros(len(val_rows), bool)
        for s in range(0, len(val_rows), 500):
            rows = val_rows[s : s + 500]
            d = lsq[None, :] - 2 * self.q_feat[rows] @ lf.T
            nn = np.argpartition(d, kmax, axis=1)[:, :kmax]
            nn = np.take_along_axis(nn, np.argsort(np.take_along_axis(d, nn, 1), 1), 1)
            for i, r in enumerate(rows):
                pq = self.q_anchor[r]
                y = self.sig[r, H : H + F]
                preds = []
                for j in nn[i]:
                    start = lanc[j] + (H - pq)
                    preds.append(self.sig[lrec[j], start : start + F] - lmean[j] + self.q_mean[r])
                same[s + i] = self.pid[lrec[nn[i, 0]]] == self.pid[r]
                for k in k_list:
                    err[k][s + i] = np.abs(np.mean(preds[:k], 0) - y).mean()
        out = {"n_val_queries": len(val_rows), "same_patient_top1": float(same.mean())}
        for k in k_list:
            out[f"mae_k{k}"] = float(pd.Series(err[k]).groupby(self.pid[val_rows]).mean().mean())
        train_pat = set(self.pid[train_rows])
        overlap = np.array([p in train_pat for p in self.pid[val_rows]])
        out["val_records_with_patient_in_train"] = float(overlap.mean())
        if overlap.any():
            out["mae_k1_overlap_records"] = float(err[1][overlap].mean())
            out["mae_k1_non_overlap_records"] = float(err[1][~overlap].mean())
        return out


def balance(meta: pd.DataFrame, tr: np.ndarray, va: np.ndarray) -> float:
    """Max |SMD| over superclasses and device families between train and validation."""
    dev = meta.device.str.split().str[0]
    cols = {f"sc_{c}": meta[f"sc_{c}"] for c in SUPERCLASSES}
    cols.update({f"dev_{d}": dev == d for d in dev.unique()})
    worst = 0.0
    for v in cols.values():
        a, b = v.to_numpy()[va].astype(float), v.to_numpy()[tr].astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2)
        worst = max(worst, abs(a.mean() - b.mean()) / sd if sd > 0 else 0)
    return float(worst)


def record_split(n: int, frac: float, seed: int, strata: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    if strata is None:
        perm = rng.permutation(n)
        cut = round(frac * n)
        return np.sort(perm[cut:]), np.sort(perm[:cut])
    val = []
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        val.append(idx[: round(frac * len(idx))])
    val = np.sort(np.concatenate(val))
    return np.setdiff1d(np.arange(n), val), val


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    S: dict = {}

    # ------------------------------------------------------------ H4.1 structural
    ids = fold_assignments(cfg.paths.data)
    part = ids.strat_fold.map(lambda f: "train" if f <= 8 else "val" if f == 9 else "test")
    pat_parts = ids.groupby("patient_id").apply(
        lambda g: part[g.index].nunique(), include_groups=False
    )
    meta9 = load_metadata(cfg.paths.data, range(1, 10))
    multi = meta9[meta9.patient_id.duplicated(keep=False)].sort_values(
        ["patient_id", "recording_dt"]
    )
    sex_const = multi.groupby("patient_id").sex.nunique().eq(1)
    g = multi[~multi.age_over_89].groupby("patient_id")
    pairs = pd.DataFrame(
        {"d_age": g.age.diff(), "d_years": g.recording_dt.diff().dt.days / 365.25}
    ).dropna()
    consistent = (pairs.d_age - pairs.d_years).abs() <= 1.5
    S["H4_1"] = {
        "records_all_folds": len(ids),
        "ecg_id_unique": bool(ids.index.is_unique),
        "patients_in_more_than_one_partition": int((pat_parts > 1).sum()),
        "patients_train_val_test": {
            p: int(ids[part == p].patient_id.nunique()) for p in ("train", "val", "test")
        },
        "multi_record_patients_folds1_9": int(multi.patient_id.nunique()),
        "sex_constant_share": float(sex_const.mean()),
        "age_vs_date_consistent_share": float(consistent.mean()),
        "age_vs_date_pairs": len(pairs),
    }
    S["H4_1"]["pass"] = (
        S["H4_1"]["ecg_id_unique"]
        and S["H4_1"]["patients_in_more_than_one_partition"] == 0
        and S["H4_1"]["sex_constant_share"] >= 0.99
        and S["H4_1"]["age_vs_date_consistent_share"] >= 0.95
    )

    # ------------------------------------------------------------ data + peaks
    meta, sig, _ = load_dataset(cfg, "dev", cache_dir=".cache/data")
    sig = np.ascontiguousarray(sig, dtype=np.float32)
    with Pool(38) as pool:
        peaks = pool.map(peaks_pair, [s[:, 1] for s in sig], chunksize=64)
    hist_vs_full = [
        not np.array_equal(h[h < H - 30], f[f < H - 30]) for h, f in peaks
    ]  # history peaks (away from the boundary) differ when detected on the full record
    S["peak_detection_noncausal_share"] = float(np.mean(hist_vs_full))

    # ------------------------------------------------------------ H4.2 split comparison
    probe = Probe(meta, sig, peaks)
    S["probe"] = {"library_anchors_total": int(len(probe.lib_rec)),
                  "query_coverage": float((probe.q_anchor >= 0).mean())}  # fmt: skip
    n = len(meta)
    strata = (
        meta[[f"sc_{c}" for c in SUPERCLASSES]]
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
        .to_numpy()
    )
    rows = []

    def run(strategy, seed, tr, va):
        res = probe.evaluate(tr, va)
        res.update(strategy=strategy, seed=seed, balance_max_abs_smd=balance(meta, tr, va))
        rows.append(res)
        print(
            strategy,
            seed,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()},
            flush=True,
        )

    pos = {e: i for i, e in enumerate(meta.index)}
    to_rows = lambda frame: np.array([pos[e] for e in frame.index])  # noqa: E731
    for seed in SEEDS:
        run("R random record", seed, *record_split(n, 0.2, seed))
        run("SR stratified record", seed, *record_split(n, 0.2, seed, strata))
        p = patient_holdout(meta, 0.2, seed)
        run("P patient holdout", seed, to_rows(p["train"]), to_rows(p["val"]))
    of = official_split(meta, cfg)
    run("OF official folds", 0, to_rows(of["train"]), to_rows(of["val"]))
    rng = np.random.default_rng(42)
    fold = rng.permutation(np.arange(n) % 5)
    for f in range(5):
        run("KF record 5-fold", f, np.flatnonzero(fold != f), np.flatnonzero(fold == f))
    for f, parts in enumerate(group_kfold(meta, 5, 42)):
        run("GKF patient 5-fold", f, to_rows(parts["train"]), to_rows(parts["val"]))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "table_split_comparison.csv", index=False, float_format="%.5f")
    summary = res.groupby("strategy").agg(
        mae_k1=("mae_k1", "mean"), mae_k1_sd=("mae_k1", "std"), mae_k5=("mae_k5", "mean"),
        same_patient_top1=("same_patient_top1", "mean"),
        val_records_with_patient_in_train=("val_records_with_patient_in_train", "mean"),
        balance_max_abs_smd=("balance_max_abs_smd", "mean"), runs=("seed", "size"),
    )  # fmt: skip
    summary.to_csv(OUT / "table_split_summary.csv", float_format="%.5f")
    r = res[res.strategy.str.startswith("R ")].set_index("seed").mae_k1
    pp = res[res.strategy.str.startswith("P ")].set_index("seed").mae_k1
    diff = (pp - r).to_numpy()
    kf = res[res.strategy.str.startswith("KF")].mae_k1.to_numpy()
    gkf = res[res.strategy.str.startswith("GKF")].mae_k1.to_numpy()
    S["H4_2"] = {
        "mae_k1_patient_minus_record_by_seed": diff.tolist(),
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
        "relative_optimism_record_split": float(diff.mean() / pp.mean()),
        "kfold_mean_diff_gkf_minus_kf": float(gkf.mean() - kf.mean()),
        "pass": bool(diff.mean() > 2 * diff.std(ddof=1) and diff.mean() > 0),
    }

    # ------------------------------------------------------------ H4.3 metadata leakage
    hm = []
    for (hist_p, full_p), (_, row) in zip(peaks, meta.iterrows(), strict=True):
        rr_h = np.diff(hist_p)
        if len(rr_h) < 3:
            continue
        med = np.median(rr_h)
        fut = full_p[full_p >= H - 1]
        prev = full_p[full_p < H - 1]
        seq = np.concatenate([prev[-1:], fut])
        rr_f = np.diff(seq)
        hm.append({
            "hist_regular": bool(np.all(np.abs(rr_h - med) <= 0.2 * med)),
            "fut_irregular": bool(len(rr_f) and np.any(np.abs(rr_f - med) > 0.2 * med)),
            "annotated": bool({"PVC", "PAC"} & set(row.scp) or pd.notna(row.extra_beats)),
        })  # fmt: skip
    hm = pd.DataFrame(hm)
    reg = hm[hm.hist_regular]
    rate = reg.groupby("fut_irregular").annotated.agg(["mean", "size"])
    ratio = rate.loc[True, "mean"] / rate.loc[False, "mean"]
    S["H4_3"] = {
        "history_regular_records": len(reg),
        "annotation_rate_future_irregular": float(rate.loc[True, "mean"]),
        "n_future_irregular": int(rate.loc[True, "size"]),
        "annotation_rate_future_regular": float(rate.loc[False, "mean"]),
        "n_future_regular": int(rate.loc[False, "size"]),
        "ratio": float(ratio),
        "annotation_rate_history_irregular": float(hm[~hm.hist_regular].annotated.mean()),
        "pass": bool(ratio >= 2),
    }

    # ------------------------------------------------------------ H4.4 preprocessing leakage
    sos = sps.butter(2, 0.67, "lowpass", fs=100, output="sos")
    zi = sps.sosfilt_zi(sos)
    target_mean = sig[:, H : H + F].mean(1)
    zero_phase = sps.sosfiltfilt(sos, sig, axis=1)[:, H - 1]
    causal = np.stack(
        [sps.sosfilt(sos, x, axis=0, zi=zi[:, :, None] * x[0])[0][H - 1] for x in sig]
    )
    preds = {"zero-phase low-pass at 6.99 s (non-causal)": zero_phase,
             "causal low-pass at 6.99 s": causal,
             "history mean": sig[:, :H].mean(1),
             "mean of last 1 s of history": sig[:, H - 100 : H].mean(1)}  # fmt: skip
    mae = {k: float(np.abs(v - target_mean).mean()) for k, v in preds.items()}
    rel = 1 - mae["zero-phase low-pass at 6.99 s (non-causal)"] / mae["causal low-pass at 6.99 s"]
    rec = sig[0].astype(np.float64)
    transforms = {
        "zero-phase 0.67 Hz low-pass (sosfiltfilt)": lambda x: sps.sosfiltfilt(sos, x, axis=0),
        "causal 0.67 Hz low-pass (sosfilt)": lambda x: sps.sosfilt(sos, x, axis=0),
        "whole-record z-score": lambda x: (x - x.mean(0)) / x.std(0),
        "history-only z-score": lambda x: (x - x[:H].mean(0)) / x[:H].std(0),
        "NeuroKit2 ecg_clean (lead II)": lambda x: __import__("neurokit2").ecg_clean(
            x[:, 1], sampling_rate=100
        ),
    }
    S["H4_4"] = {
        "target_mean_mae_mV": mae,
        "zero_phase_improvement_over_causal": float(rel),
        "pass": bool(rel >= 0.10),
        "causality_violation_mV_example_record": {
            k: causality_violation(f, rec, H) for k, f in transforms.items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=str) + "\n")
    print(json.dumps(S, indent=2, default=str))
    figures(res, summary, S, pairs, sig, sos, zi)


def figures(res, summary, S, pairs, sig, sos, zi):
    order = ["R random record", "SR stratified record", "KF record 5-fold", "P patient holdout",
             "GKF patient 5-fold", "OF official folds"]  # fmt: skip
    sm = summary.reindex(order)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    colors = ["#c44e52"] * 3 + ["#55a868"] * 3
    for i, s in enumerate(order):
        v = res[res.strategy == s].mae_k1
        ax[0].scatter([i] * len(v), v, color=colors[i], s=18)
    ax[0].set_xticks(range(len(order)), [o.split(" ", 1)[0] for o in order])
    ax[0].set(ylabel="kNN (k=1) patient-macro MAE (mV)", title="Memorization probe error by split")
    ax[1].bar(range(len(order)), 100 * sm.same_patient_top1, color=colors)
    ax[1].set_xticks(range(len(order)), [o.split(" ", 1)[0] for o in order])
    ax[1].set(
        ylabel="% val queries whose nearest neighbour is the same patient",
        title="Patient memorization",
    )
    ax[2].bar(range(len(order)), sm.balance_max_abs_smd, color=colors)
    ax[2].axhline(0.1, color="k", ls="--", lw=0.8)
    ax[2].set_xticks(range(len(order)), [o.split(" ", 1)[0] for o in order])
    ax[2].set(ylabel="max |SMD| superclass/device (train vs val)", title="Label and device balance")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_split_comparison.png", dpi=140)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    h3 = S["H4_3"]
    ax[0].bar(["history irregular", "history regular,\nfuture regular", "history regular,\nfuture irregular"],
              100 * np.array([h3["annotation_rate_history_irregular"], h3["annotation_rate_future_regular"],
                              h3["annotation_rate_future_irregular"]]), color=["#8172b2", "#4c72b0", "#c44e52"])  # fmt: skip
    ax[0].set(
        ylabel="% with PVC/PAC/extra-beat annotation", title="Labels describe the forecast window"
    )
    m = S["H4_4"]["target_mean_mae_mV"]
    ax[1].barh(list(m), list(m.values()), color=["#c44e52", "#55a868", "#4c72b0", "#4c72b0"])
    ax[1].set(
        xlabel="MAE predicting the target-window mean (mV)",
        title="Non-causal filtering leaks the future",
    )
    ax[1].tick_params(axis="y", labelsize=8)
    x = sig[int(np.argmax(np.abs(sig[:, 690:760, 1]).max(1)))][:, 1].astype(np.float64)
    t = np.arange(len(x)) / 100
    ax[2].plot(t, x, color="#bbbbbb", lw=0.8, label="lead II")
    ax[2].plot(t, sps.sosfiltfilt(sos, x), color="#c44e52", label="zero-phase low-pass")
    ax[2].plot(t, sps.sosfilt(sos, x, zi=zi * x[0])[0], color="#55a868", label="causal low-pass")
    ax[2].axvline(7, color="k", ls="--", lw=0.8)
    ax[2].set(
        xlim=(5, 9.9),
        xlabel="time (s)",
        ylabel="mV",
        title="Filter output near the forecast origin",
    )
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_metadata_and_preprocessing.png", dpi=140)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(pairs.d_years, pairs.d_age, s=4, alpha=0.4)
    lim = [0, max(pairs.d_years.max(), pairs.d_age.max()) + 1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set(xlabel="years between recordings", ylabel="age difference (years)",
           title=f"Patient identity check: {100 * S['H4_1']['age_vs_date_consistent_share']:.1f}% consistent")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig3_identity.png", dpi=140)


if __name__ == "__main__":
    main()
