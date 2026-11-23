"""Phase 5 summary figure: paired differences vs the default pipeline (ridge and CNN tiers).

python research/phase5_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("reports/phase5_preprocessing")
METRICS = [("mae", "ΔMAE (mV)"), ("dtw_mae", "ΔDTW-MAE (mV)"), ("template_mae", "ΔTemplate MAE (mV)"),
           ("beat_timing_ms", "ΔBeat timing (ms)")]  # fmt: skip


def panel(ax_row, table: pd.DataFrame, label_col: str, title: str) -> None:
    t = table[table[label_col] != "default"].dropna(subset=["d_mae"]) if "d_mae" in table else table
    t = t.iloc[::-1]
    for ax, (m, xlabel) in zip(ax_row, METRICS, strict=True):
        lo, hi, d = t[f"d_{m}_lo"], t[f"d_{m}_hi"], t[f"d_{m}"]
        better = hi < 0
        worse = lo > 0
        colors = [
            "#2ca02c" if b else "#d62728" if w else "#7f7f7f"
            for b, w in zip(better, worse, strict=True)
        ]
        for i, (di, lo_i, hi_i, c) in enumerate(zip(d, lo, hi, colors, strict=True)):
            ax.plot([lo_i, hi_i], [i, i], color=c, lw=1.2)
            ax.scatter([di], [i], color=c, s=14, zorder=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(t)), t[label_col] if ax is ax_row[0] else [""] * len(t), fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    ax_row[0].set_title(title, loc="left", fontsize=10)


def main() -> None:
    ridge = pd.read_csv(OUT / "table_ridge.csv")
    ridge["label"] = ridge.category + ": " + ridge.variant
    cnn = pd.concat(
        [
            pd.read_csv(OUT / f"table_cnn_{s}.csv")
            for s in ("shortlist", "augment")
            if (OUT / f"table_cnn_{s}.csv").exists()
        ]
    ).drop_duplicates("variant")
    n_r, n_c = len(ridge) - 1, len(cnn) - 1
    fig, ax = plt.subplots(
        2, 4, figsize=(17, 0.28 * (n_r + n_c) + 3), gridspec_kw={"height_ratios": [n_r, n_c]}
    )
    panel(
        ax[0],
        ridge,
        "label",
        "Ridge tier (vs default; green = better, red = worse, 95% patient-cluster CI)",
    )
    panel(ax[1], cnn, "variant", "CNN tier, 3-seed ensembles (vs default)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_effects_vs_default.png", dpi=140)


if __name__ == "__main__":
    main()
