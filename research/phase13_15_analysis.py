"""Phases 13-15: physiological plausibility and error analysis on the out-of-fold (OOF)
forecasts of folds 1-9 (PREREGISTRATION.md in each report folder).

    python research/phase13_15_analysis.py   # after phase12_robustness.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import phase0_feasibility as p0  # noqa: E402
import phase10_objectives as p10  # noqa: E402
import phase11_tuning as p11  # noqa: E402
import phase12_robustness as p12  # noqa: E402
from phase5_cnn import limb_residual  # noqa: E402

OUT13, OUT14, OUT15 = (
    Path(f"reports/{p}")
    for p in ("phase13_plausibility", "phase14_patient_errors", "phase15_signal_errors")
)
H, F, FS = 700, 290, 100
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUITE = p12.SUITE
N_BOOT = 1000


def load_oof():
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset

    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    preds = {a: np.zeros((len(meta), F, 12), np.float32) for a in ("final", "l1")}
    paths = p12.oof_paths()
    for k in range(1, 10):
        rows = np.flatnonzero(meta.strat_fold == k)
        for a in preds:
            preds[a][rows] = np.mean([np.load(p)["pred"] for p in paths[a][k]], axis=0)
    preds["sn4"] = np.load(p10.SN4)
    preds["flat"] = np.broadcast_to(
        np.median(sig[:, :H], axis=1, keepdims=True), (len(meta), F, 12)
    ).astype(np.float32)
    t = pd.read_parquet(p12.CACHE / "oof_scores.parquet")
    s = pd.read_parquet(p12.CACHE / "strata.parquet")
    return meta, sig, preds, t, s


def boot_ci(values, pid, seed=0):
    from ptbxl_forecast.stats import cluster_bootstrap

    return cluster_bootstrap(values, pid, N_BOOT, seed)


# ------------------------------------------------------------------ Phase 13
def hf_share(x: np.ndarray, cutoff: float = 15.0) -> np.ndarray:
    """Share of (mean-removed) power above ``cutoff`` Hz, lead II, per record."""
    x = x[:, :, 1] - x[:, :, 1].mean(axis=1, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], 1 / FS)
    return p[:, f > cutoff].sum(1) / np.maximum(p.sum(1), 1e-12)


def phase13(meta, sig, preds, t) -> dict:
    OUT13.mkdir(parents=True, exist_ok=True)
    y, hist = sig[:, H : H + F].astype(np.float64), sig[:, :H].astype(np.float64)
    pid = meta.patient_id.to_numpy()
    wide = {a: g.set_index("ecg_id").loc[meta.index] for a, g in t.groupby("arm")}
    lo, hi = hist.min(axis=1, keepdims=True), hist.max(axis=1, keepdims=True)
    rows = []
    for a in ("truth", "final", "l1", "sn4", "flat"):
        x = y if a == "truth" else preds[a].astype(np.float64)
        r = {"arm": a}
        lr = limb_residual(x)
        r["limb_median_mV"], r["limb_share_gt_0.01"] = (
            float(np.median(lr)),
            float((lr > 0.01).mean()),
        )
        p2p = np.ptp(x[:, :, 1], axis=1) / np.maximum(np.ptp(y[:, :, 1], axis=1), 1e-6)
        r["p2p_ratio_median"] = float(np.median(p2p))
        r["p2p_ratio_iqr"] = f"{np.percentile(p2p, 25):.2f}-{np.percentile(p2p, 75):.2f}"
        r["hf_share_median"] = float(np.median(hf_share(x)))
        out = ((x < lo - 0.5) | (x > hi + 0.5)).any(axis=(1, 2))
        r["share_outside_history_range_0.5mV"] = float(out.mean())
        if a != "truth":
            w = wide[a]
            for m in ("r_amp_err", "st_err", "t_amp_err"):
                r[m] = p12_macro(w[m], pid)
            bt = w.beat_timing_ms
            for name, (l_, h_) in {
                "le25": (-1, 25),
                "25-50": (25, 50),
                "50-150": (50, 150),
                "gt150": (150, 1e9),
            }.items():
                r[f"timing_{name}"] = float(((bt > l_) & (bt <= h_)).mean())
            npb, ntb = w.n_pred_beats.to_numpy(), w.n_true_beats.to_numpy()
            r["share_no_pred_beats"] = float((npb == 0).mean())
            r["beat_count_ratio_median"] = float(np.median(npb[ntb > 0] / ntb[ntb > 0]))
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT13 / "table_plausibility.csv", index=False, float_format="%.5g")
    f = tab.set_index("arm")
    S = {
        "H13_1_limb_share_final_le_l1": bool(f.loc["final", "limb_share_gt_0.01"] <= f.loc["l1", "limb_share_gt_0.01"]),
        "H13_2_final_damped_p2p_lt_0.8": bool(f.loc["final", "p2p_ratio_median"] < 0.8),
        "H13_3_final_in_range_gt_99pct": bool(f.loc["final", "share_outside_history_range_0.5mV"] < 0.01),
    }  # fmt: skip
    json.dump(S, (OUT13 / "summary.json").open("w"), indent=2)
    # figure: lead-II p2p ratio and timing bands
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, c in (("final", "#d62728"), ("l1", "#999999"), ("sn4", "#2ca02c")):
        r = np.ptp(preds[a][:, :, 1], axis=1) / np.maximum(np.ptp(y[:, :, 1], axis=1), 1e-6)
        ax[0].hist(np.clip(r, 0, 2), 80, histtype="step", color=c, label=a, density=True)
    ax[0].axvline(1, color="k", ls="--", lw=0.8)
    ax[0].set(
        xlabel="forecast / true peak-to-peak, lead II (clipped at 2)",
        title="Amplitude of forecasts",
    )
    ax[0].legend()
    bands = ["timing_le25", "timing_25-50", "timing_50-150", "timing_gt150"]
    bottom = np.zeros(4)
    arms = ["final", "l1", "sn4", "flat"]
    for b, c in zip(bands, ["#1a9850", "#a6d96a", "#fdae61", "#d73027"], strict=True):
        v = f.loc[arms, b].to_numpy()
        ax[1].bar(arms, v, bottom=bottom, color=c, label=b.replace("timing_", "") + " ms")
        bottom += v
    ax[1].set(ylabel="share of records", title="Beat timing bands (D6.6)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT13 / "fig1_plausibility.png", dpi=130)
    print(tab.round(4).to_string(), "\n", S)
    return S


def p12_macro(v, pid) -> float:
    from ptbxl_forecast.stats import patient_macro

    return patient_macro(v, pid)


# ------------------------------------------------------------------ Phase 14
def patient_factors(meta: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    f = s[["patient_id", "device", "rhythm", "heart_rate"]].copy()
    age = meta.age_years
    f["age"] = pd.cut(
        age, [0, 40, 60, 80, 200], labels=["<40", "40-59", "60-79", "80-89"], right=False
    ).astype(str)
    f.loc[meta.age_over_89, "age"] = ">89"
    f["sex"] = meta.sex_label
    for c in ("NORM", "MI", "STTC", "CD", "HYP"):
        f[f"dx_{c}"] = np.where(meta[f"sc_{c}"], "yes", "no")
    return f


def phase14(meta, t, s) -> dict:
    OUT14.mkdir(parents=True, exist_ok=True)
    f = patient_factors(meta, s)
    wide = {a: g.set_index("ecg_id").loc[meta.index] for a, g in t.groupby("arm")}
    d_l1 = wide["final"].mae.to_numpy() - wide["l1"].mae.to_numpy()
    d_sn4_t = wide["final"].beat_timing_ms.to_numpy() - wide["sn4"].beat_timing_ms.to_numpy()
    fin = wide["final"]
    rows = []
    for fac in [
        "age",
        "sex",
        "device",
        "rhythm",
        "heart_rate",
        *[c for c in f if c.startswith("dx_")],
    ]:
        for lev, ids in f.groupby(fac).groups.items():
            m = meta.index.get_indexer(pd.Index(ids))
            pid = f.patient_id.to_numpy()[m]
            r = {"factor": fac, "level": lev, "records": len(m), "patients": len(np.unique(pid))}
            for metric in SUITE:
                r[f"final_{metric}"], r[f"final_{metric}_lo"], r[f"final_{metric}_hi"] = boot_ci(
                    fin[metric].to_numpy()[m], pid
                )
            r["d_mae_vs_l1_uV"], lo, hi = (v * 1000 for v in boot_ci(d_l1[m], pid))
            r["d_mae_vs_l1_ci"], r["d_mae_vs_l1_hi"] = f"[{lo:+.1f}, {hi:+.1f}]", hi
            r["d_timing_vs_sn4_ms"], lo, hi = boot_ci(d_sn4_t[m], pid)
            r["d_timing_vs_sn4_ci"] = f"[{lo:+.1f}, {hi:+.1f}]"
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab = tab[(tab.level != "nan") & (tab.level != "None")]
    tab.to_csv(OUT14 / "table_factors.csv", index=False, float_format="%.5g")
    reg = regression(f, fin.mae.to_numpy(), meta)
    reg.to_csv(OUT14 / "table_regression_log_mae.csv", index=False, float_format="%.4g")
    sig_terms = reg[((reg.lo > 0) | (reg.hi < 0)) & (reg.term != "intercept")]
    top = sig_terms.reindex(sig_terms.coef.abs().sort_values(ascending=False).index).head(6)
    S = {"largest_adjusted_effects_pct": {r.term: round(100 * (np.exp(r.coef) - 1), 1) for r in top.itertuples()},
         "factor_levels_where_final_not_better_than_l1": tab[tab.d_mae_vs_l1_hi >= 0][["factor", "level"]].values.tolist()}  # fmt: skip
    json.dump(S, (OUT14 / "summary.json").open("w"), indent=2)
    fig, ax = plt.subplots(figsize=(9, 9))
    sub = tab[~tab.factor.str.startswith("dx_") | (tab.level == "yes")]
    y = np.arange(len(sub))
    ax.errorbar(
        sub.final_mae,
        y,
        xerr=[sub.final_mae - sub.final_mae_lo, sub.final_mae_hi - sub.final_mae],
        fmt="o",
        color="#d62728",
    )
    ax.set_yticks(
        y,
        sub.factor.str.replace("dx_", "dx: ")
        + " = "
        + sub.level.astype(str)
        + " (n="
        + sub.patients.astype(str)
        + ")",
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="patient-macro MAE (mV), 95% CI",
        title="Final model, OOF folds 1-9: MAE by patient factor",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT14 / "fig1_factors.png", dpi=130)
    print(
        tab[
            [
                "factor",
                "level",
                "patients",
                "final_mae",
                "d_mae_vs_l1_uV",
                "d_mae_vs_l1_ci",
                "final_beat_timing_ms",
                "d_timing_vs_sn4_ms",
                "d_timing_vs_sn4_ci",
            ]
        ]
        .round(4)
        .to_string()
    )
    print(reg.round(4).to_string(), "\n", S)
    return S


def regression(f: pd.DataFrame, mae: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """OLS of log MAE on all factors jointly (reference levels dropped), with
    patient-cluster bootstrap 95% CIs. Adds baseline wander (log) as a covariate."""
    p3 = pd.read_csv(".cache/data/phase3_per_record.csv", index_col="ecg_id")
    X = pd.get_dummies(
        f[
            ["age", "sex", "device", "rhythm", "heart_rate", *[c for c in f if c.startswith("dx_")]]
        ].astype(str),
        drop_first=False,
    )
    ref = [
        "age_60-79",
        "sex_male",
        "device_CS100",
        "rhythm_sinus",
        "heart_rate_60-100",
        *[f"dx_{c}_no" for c in ("NORM", "MI", "STTC", "CD", "HYP")],
    ]
    X = X.drop(columns=[c for c in X if c in ref or c.endswith("_nan") or c.endswith("_None")])
    X["log_baseline_wander"] = np.log(p3.bw_pp_med.reindex(meta.index).to_numpy() + 1e-3)
    X.insert(0, "intercept", 1.0)
    A = X.to_numpy(float)
    ok = np.isfinite(A).all(1) & np.isfinite(mae) & (mae > 0)
    A, yv, pid = A[ok], np.log(mae[ok]), f.patient_id.to_numpy()[ok]
    coef = np.linalg.lstsq(A, yv, rcond=None)[0]
    codes, uniq = pd.factorize(pid)
    groups = pd.Series(np.arange(len(codes))).groupby(codes).apply(np.array).to_list()
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots.append(np.linalg.lstsq(A[idx], yv[idx], rcond=None)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame(
        {
            "term": X.columns,
            "coef": coef,
            "lo": lo,
            "hi": hi,
            "pct_effect": 100 * (np.exp(coef) - 1),
        }
    )


# ------------------------------------------------------------------ Phase 15
def phase15(meta, sig, preds, t, s) -> dict:
    OUT15.mkdir(parents=True, exist_ok=True)
    y = sig[:, H : H + F].astype(np.float64)
    pid = meta.patient_id.to_numpy()
    from ptbxl_forecast.stats import patient_macro

    # per lead and per horizon
    lead_rows, hz = [], {}
    for a in ("final", "l1", "sn4", "flat"):
        err = np.abs(preds[a].astype(np.float64) - y)
        lead_rows.append(
            {"arm": a, **{L: patient_macro(err[:, :, j].mean(1), pid) for j, L in enumerate(LEADS)}}
        )
        hz[a] = err.mean(axis=(0, 2))
    leads = pd.DataFrame(lead_rows)
    leads.to_csv(OUT15 / "table_per_lead.csv", index=False, float_format="%.5g")
    hzt = pd.DataFrame({a: v.reshape(29, 10).mean(1) for a, v in hz.items()})
    hzt.insert(0, "horizon_ms_end", np.arange(100, 2901, 100))
    hzt.to_csv(OUT15 / "table_per_horizon.csv", index=False, float_format="%.5g")
    # periodicity of the history (ACF peak of the sn4 period estimate)
    acf_path = p12.CACHE / "acf_peak.npy"
    if not acf_path.exists():
        from multiprocessing import Pool

        with Pool(38) as pool:
            np.save(
                acf_path,
                np.array(pool.map(_acf, [sig[i, :H] for i in range(len(sig))], chunksize=64)),
            )
    acf = np.load(acf_path)
    wide = {a: g.set_index("ecg_id").loc[meta.index] for a, g in t.groupby("arm")}
    q = pd.qcut(acf, 5, labels=["Q1 (least periodic)", "Q2", "Q3", "Q4", "Q5 (most periodic)"])
    per_rows = []
    for lev in q.categories:
        m = np.asarray(q == lev)
        r = {
            "acf_quintile": lev,
            "acf_range": f"{acf[m].min():.2f}-{acf[m].max():.2f}",
            "records": int(m.sum()),
        }
        for a in ("final", "l1", "sn4", "flat"):
            for metric in ("mae", "beat_timing_ms"):
                r[f"{a}_{metric}"] = patient_macro(wide[a][metric].to_numpy()[m], pid[m])
        per_rows.append(r)
    per = pd.DataFrame(per_rows)
    per.to_csv(OUT15 / "table_periodicity.csv", index=False, float_format="%.5g")
    # gate behaviour on fold 9 (saved models)
    gates = gate_analysis(meta, sig, acf)
    # worst 1% records of the final model
    mae = wide["final"].mae.to_numpy()
    worst = mae >= np.quantile(mae, 0.99)
    f = s.assign(q_any=meta.q_any.to_numpy(), acf=acf)
    comp = []
    for col, val in (("q_any", True), ("rhythm", "AF/flutter"), ("rhythm", "paced"), ("baseline_wander", "high"),
                     ("device", "CS100"), ("device", "AT-6"), ("device", "CS-12"), ("device", "AT-60")):  # fmt: skip
        v = (f[col] == val).to_numpy()
        comp.append(
            {
                "feature": f"{col}={val}",
                "share_worst_1pct": float(v[worst].mean()),
                "share_rest": float(v[~worst].mean()),
            }
        )
    comp.append(
        {
            "feature": "median ACF peak",
            "share_worst_1pct": float(np.median(acf[worst])),
            "share_rest": float(np.median(acf[~worst])),
        }
    )
    comp.append({"feature": "median true lead-II p2p (mV)", "share_worst_1pct": float(np.median(np.ptp(y[worst, :, 1], 1))),
                 "share_rest": float(np.median(np.ptp(y[~worst, :, 1], 1)))})  # fmt: skip
    worst_t = pd.DataFrame(comp)
    worst_t.to_csv(OUT15 / "table_worst_1pct.csv", index=False, float_format="%.4g")
    figures15(leads, hzt, per, gates)
    best_gain = per.assign(g=per.l1_mae - per.final_mae).sort_values("g")
    S = {
        "worst_lead_final": leads.set_index("arm").loc["final"].idxmax(),
        "best_lead_final": leads.set_index("arm").loc["final"].idxmin(),
        "final_mae_first_vs_last_100ms": [float(hzt.final.iloc[0]), float(hzt.final.iloc[-1])],
        "gain_vs_l1_uV_by_acf_quintile": {r.acf_quintile: round(1000 * (r.l1_mae - r.final_mae), 2) for r in per.itertuples()},
        "gain_largest_in": best_gain.acf_quintile.iloc[-1],
        "gate": gates["summary"],
    }  # fmt: skip
    json.dump(S, (OUT15 / "summary.json").open("w"), indent=2, default=float)
    print(
        leads.round(4).to_string(),
        "\n",
        hzt.round(4).to_string(),
        "\n",
        per.round(4).to_string(),
        "\n",
        worst_t.round(3).to_string(),
        "\n",
        S,
    )
    return S


def _acf(x) -> float:
    hist = x.astype(np.float64)
    return p0.estimate_period(hist, p0.rpeaks(hist[:, 1]))[1]


def gate_analysis(meta, sig, acf) -> dict:
    """Mean gate value of the 6 saved fold-9 models: per horizon step and per record."""
    import torch

    from ptbxl_forecast.preprocessing import pack_baseline

    c = p11.final_config()
    rows9 = np.flatnonzero(meta.strat_fold == 9)
    sn4 = np.load(p10.SN4)[rows9]
    packed = pack_baseline(sig[rows9].astype(np.float64), sn4.astype(np.float64), H)[:, :H]
    if c["model"] != "gated":
        return {"summary": "final model has no gate"}
    from ptbxl_forecast.models import GatedBaselineCNNLSTM

    gates = []
    torch.set_num_threads(8)
    for s in p12.SEEDS6:
        path = p12.keep_path(s)
        pipe = pickle.load(path.with_suffix(".pipe.pkl").open("rb"))
        model = GatedBaselineCNNLSTM(pipe.in_channels, 12, F, c["width"], c["hidden"], c["layers"])
        model.load_state_dict(torch.load(path.with_suffix(".pt")))
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(packed), 512):
                x = torch.from_numpy(pipe.inputs(packed[i : i + 512]).astype(np.float32))
                b = model.body
                z = b.stem(x[..., : model.in_channels].transpose(1, 2)).transpose(1, 2)
                h = b.lstm(z)[0][:, -1]
                out.append(torch.sigmoid(model.gate(h)).view(-1, F, 12).numpy())
        gates.append(np.concatenate(out))
    g = np.mean(gates, axis=0)  # (n, F, 12)
    rec = g.mean(axis=(1, 2))
    sn4_mae = np.abs(sn4 - sig[rows9, H : H + F]).mean(axis=(1, 2))
    from scipy.stats import spearmanr

    summ = {"mean_gate": float(g.mean()), "gate_first_100ms": float(g[:, :10].mean()), "gate_last_100ms": float(g[:, -10:].mean()),
            "spearman_gate_vs_acf": float(spearmanr(rec, acf[rows9]).statistic),
            "spearman_gate_vs_sn4_mae": float(spearmanr(rec, sn4_mae).statistic),
            "gate_per_lead": dict(zip(LEADS, g.mean(axis=(0, 1)).round(3).tolist(), strict=True))}  # fmt: skip
    return {"summary": summ, "per_step": g.mean(axis=(0, 2)), "rec": rec, "acf": acf[rows9]}


def figures15(leads, hzt, per, gates) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
    L = leads.set_index("arm")
    x = np.arange(12)
    for i, (a, c) in enumerate(
        (("final", "#d62728"), ("l1", "#999999"), ("sn4", "#2ca02c"), ("flat", "k"))
    ):
        ax[0].bar(x + (i - 1.5) * 0.2, L.loc[a, LEADS], 0.2, color=c, label=a)
    ax[0].set_xticks(x, LEADS)
    ax[0].set(ylabel="patient-macro MAE (mV)", title="MAE per lead (OOF)")
    ax[0].legend(fontsize=8)
    for a, c in (("final", "#d62728"), ("l1", "#999999"), ("sn4", "#2ca02c"), ("flat", "k")):
        ax[1].plot(hzt.horizon_ms_end / 1000, hzt[a], color=c, marker=".", label=a)
        ax[2].plot(np.arange(5), per[f"{a}_mae"], color=c, marker="o", label=a)
    ax[1].set(
        xlabel="forecast horizon (s)", ylabel="MAE (mV)", title="MAE vs horizon (100 ms bins)"
    )
    ax[2].set_xticks(np.arange(5), ["Q1\nleast", "Q2", "Q3", "Q4", "Q5\nmost"])
    ax[2].set(
        xlabel="history periodicity (ACF peak quintile)",
        ylabel="MAE (mV)",
        title="MAE vs periodicity",
    )
    ax[1].legend(fontsize=8)
    if "per_step" in gates:
        ax[3].plot(np.arange(F) / FS, gates["per_step"], color="#d62728")
        ax[3].set(
            xlabel="forecast horizon (s)",
            ylabel="mean gate σ(g)",
            title="Trust in sn4 (fold 9, 6 models)",
            ylim=(0, 1),
        )
    fig.tight_layout()
    fig.savefig(OUT15 / "fig1_signal_errors.png", dpi=120)


def main() -> None:
    meta, sig, preds, t, s = load_oof()
    S = {
        "phase13": phase13(meta, sig, preds, t),
        "phase14": phase14(meta, t, s),
        "phase15": phase15(meta, sig, preds, t, s),
    }
    print(json.dumps(S, indent=2, default=str))


if __name__ == "__main__":
    main()
