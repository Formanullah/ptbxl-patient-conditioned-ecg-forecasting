"""Figures for the Phase 0 feasibility report (reads outputs of phase0_feasibility.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import phase0_feasibility as p0  # noqa: E402

COLORS = {
    "zero": "#999999",
    "median": "#555555",
    "mean": "#000000",
    "sn1": "#9ecae1",
    "sn4": "#2171b5",
    "sn4_trueperiod": "#6a51a3",
    "oracle": "#d94801",
    "last": "#bbbbbb",
}
LABELS = {
    "zero": "zero",
    "median": "history median",
    "mean": "history mean",
    "sn1": "seasonal naive (1 cycle)",
    "sn4": "seasonal naive (4 cycles)",
    "sn4_trueperiod": "sn4 + true future period (oracle)",
    "oracle": "history beats at true beat times (oracle)",
    "last": "persistence",
}


def fig_horizon(out: Path) -> None:
    c = pd.read_csv(out / "table_horizon_mae.csv", index_col=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in ("zero", "median", "mean", "sn1", "sn4", "sn4_trueperiod", "oracle"):
        ax[0].plot(
            c.index,
            c[name].rolling(5, center=True, min_periods=1).mean(),
            color=COLORS[name],
            label=LABELS[name],
            lw=1.6,
        )
    ax[0].set(xlabel="lead time (s)", ylabel="MAE (mV), 12 leads", title="Error vs lead time")
    ax[0].legend(fontsize=7.5)
    for name in ("sn4", "sn4_trueperiod", "oracle"):
        sk = (1 - c[name] / c["median"]).rolling(10, center=True, min_periods=1).mean()
        ax[1].plot(c.index, sk, color=COLORS[name], label=LABELS[name], lw=1.6)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(
        xlabel="lead time (s)",
        ylabel="skill vs history median",
        title="Skill = 1 − MAE / MAE(median)",
    )
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig1_horizon.png", dpi=150)


def fig_rhythm(out: Path) -> None:
    s = pd.read_csv(out / "table_skill_by_rhythm_horizon.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col in s.columns:
        ax.plot(s.index, s[col].rolling(10, center=True, min_periods=1).mean(), label=col, lw=1.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(
        xlabel="lead time (s)",
        ylabel="sn4 skill vs history mean",
        title="Structural forecast skill by rhythm",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig2_skill_by_rhythm.png", dpi=150)


def fig_groups(out: Path) -> None:
    g = pd.read_csv(out / "table_groups.csv")
    g = g[g.n_records >= 50]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(g) + 1.2))
    for key, color, dy in (
        ("skill_sn4", COLORS["sn4"], -0.18),
        ("skill_oracle", COLORS["oracle"], 0.18),
    ):
        err = np.vstack([g[key] - g[f"{key}_lo"], g[f"{key}_hi"] - g[key]])
        ax.barh(
            y + dy,
            g[key],
            height=0.34,
            xerr=err,
            color=color,
            capsize=2,
            label="sn4" if key == "skill_sn4" else "oracle timing",
        )
    ax.set_yticks(
        y,
        [f"{k}: {n} (n={m})" for k, n, m in zip(g.kind, g.group, g.n_records, strict=True)],
        fontsize=8,
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set(xlabel="whole-horizon skill vs history mean (95% patient bootstrap CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_group_skill.png", dpi=150)


def fig_drivers(out: Path, cache: Path) -> None:
    df = pd.read_csv(cache / "per_record.csv")
    df["skill"] = 1 - df.mae_sn4 / df.mae_mean
    rho = pd.read_csv(out / "table_drivers_spearman.csv", index_col=0).spearman_rho
    feats = [
        ("period_error", "relative period error", (0, 0.3)),
        ("acf_peak", "ACF peak height", (0, 1)),
        ("rr_cv_hist", "history RR CV", (0, 0.3)),
        ("perm_entropy", "permutation entropy", (0.6, 1)),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for a, (col, label, xlim) in zip(ax, feats, strict=True):
        d = df[[col, "skill"]].dropna()
        a.hexbin(
            d[col].clip(*xlim),
            d.skill.clip(-1.5, 0.8),
            gridsize=40,
            bins="log",
            cmap="Blues",
            extent=(*xlim, -1.5, 0.8),
        )
        a.axhline(0, color="k", lw=0.8)
        a.set(xlabel=label, title=f"Spearman ρ = {rho[col]:.2f}")
    ax[0].set_ylabel("per-record sn4 skill vs mean")
    fig.tight_layout()
    fig.savefig(out / "fig4_drivers.png", dpi=150)


def fig_examples(out: Path, data: Path) -> None:
    import wfdb

    meta = p0.load_metadata(data)
    rng = np.random.default_rng(p0.SEED)
    picks = [
        ("SR", meta[meta.rh_SR & ~meta.q_any & (meta.strat_fold <= 8)]),
        ("AFIB", meta[meta.rh_AFIB & ~meta.q_any & (meta.strat_fold <= 8)]),
    ]
    fig, ax = plt.subplots(len(picks), 1, figsize=(12, 3.2 * len(picks)))
    for a, (label, pool) in zip(ax, picks, strict=True):
        eid = int(rng.choice(pool.index.to_numpy()))
        sig = wfdb.rdsamp(str(data / pool.loc[eid, "filename_lr"]))[0]
        peaks = p0.rpeaks(sig[:, 1])
        preds, *_ = p0.forecasts(sig, p0.H, p0.F, peaks)
        t = np.arange(p0.N_TOTAL) / p0.FS
        a.plot(t[400:], sig[400:, 1], color="k", lw=1.2, label="observed / true future")
        tf = t[p0.H :]
        for name in ("median", "sn4", "oracle"):
            a.plot(tf, preds[name][:, 1], color=COLORS[name], lw=1.2, label=LABELS[name])
        a.axvline(p0.H / p0.FS, color="r", ls="--", lw=0.8)
        a.set(title=f"{label} example, ecg_id {eid}, lead II", xlabel="time (s)", ylabel="mV")
        a.legend(fontsize=7.5, loc="upper left", ncol=4)
    fig.tight_layout()
    fig.savefig(out / "fig5_examples.png", dpi=150)


def fig_splits(out: Path) -> None:
    s = pd.read_csv(out / "table_split_curves.csv")
    c = pd.read_csv(out / "table_context.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for col, lab in (("sn4_7to3", "7→3"), ("sn4_8to2", "8→2"), ("sn4_9to1", "9→1")):
        ax[0].plot(s.lead_time_s, s[col].rolling(5, center=True, min_periods=1).mean(), label=lab)
    ax[0].set(
        xlabel="lead time (s)", ylabel="sn4 MAE (mV)", title="Error by lead time for each split"
    )
    ax[0].legend()
    w = 0.35
    x = np.arange(len(c))
    ax[1].bar(x - w / 2, c.mae_sn4, w, label="sn4", color=COLORS["sn4"])
    ax[1].bar(x + w / 2, c.mae_mean, w, label="history mean", color=COLORS["mean"])
    ax[1].set_xticks(x, [f"{v:g} s" for v in c.context_s])
    ax[1].set(
        xlabel="context length",
        ylabel="MAE (mV) on fixed 9–10 s window",
        title="Effect of context length",
    )
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "fig6_splits_context.png", dpi=150)


def fig_metrics(out: Path) -> None:
    h = pd.read_csv(out / "table_headline.csv")
    h = h[h.dtw_point.notna()]
    x = np.arange(len(h))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(x - 0.2, h.mae_point, 0.4, label="MAE", color="#4d4d4d")
    ax[0].bar(x + 0.2, h.dtw_point, 0.4, label="DTW (±100 ms band)", color="#4393c3")
    ax[0].set_xticks(x, h.forecast, rotation=30, fontsize=8)
    ax[0].set(ylabel="mV", title="Pointwise vs timing-tolerant error")
    ax[0].legend()
    a = pd.read_csv(out / "table_blend_alpha.csv", index_col=0)
    ax[1].plot(a.index, a.alpha.rolling(10, center=True, min_periods=1).mean(), color=COLORS["sn4"])
    ax[1].set(
        xlabel="lead time (s)",
        ylabel="α(h)",
        ylim=(0, 1),
        title="MSE-optimal weight on sn4 in mean + α(sn4 − mean)\n(fitted on folds 1–8)",
    )
    fig.tight_layout()
    fig.savefig(out / "fig7_metrics_and_shrinkage.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("PTB-XL"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase0_feasibility"))
    ap.add_argument("--cache", type=Path, default=Path("results/phase0"))
    args = ap.parse_args()
    fig_horizon(args.out)
    fig_rhythm(args.out)
    fig_groups(args.out)
    fig_drivers(args.out, args.cache)
    fig_examples(args.out, args.data)
    fig_splits(args.out)
    fig_metrics(args.out)


if __name__ == "__main__":
    main()
