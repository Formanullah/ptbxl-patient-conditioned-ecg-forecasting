"""Charts for the presentation. Numbers are taken from the phase reports (cited per
chart) or recomputed from cached out-of-fold (OOF) forecasts of the final configuration
(hidden-128 sn4-gated CNN-LSTM, Phase 10 stage C runs). Fold 10 numbers come from the
Phase 17 report only.

    uv run --locked python reports/presentation/charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "research")

OUT = Path("reports/presentation/figures")
NAVY, RED, GREY, GREEN, BLACK, TEAL, ORANGE = "#1F2A44", "#C0392B", "#95A5A6", "#27AE60", "#2C3E50", "#16A085", "#E67E22"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
H, F = 700, 290
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "savefig.dpi": 170})  # fmt: skip


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


WRAP = {"CNN-LSTM": "CNN-\nLSTM", "PatchTST*": "Patch-\nTST*", "Transformer": "Trans-\nformer"}


def bars(ax, labels, values, colors, fmt="{:.3f}", title="", ylabel="", ylim=None):
    labels = [WRAP.get(str(lab), lab) for lab in labels]
    x = np.arange(len(labels))
    b = ax.bar(x, values, color=colors, width=0.65)
    for r, v in zip(b, values, strict=True):
        ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set(title=title, ylabel=ylabel)
    if ylim:
        ax.set_ylim(*ylim)


# ------------------------------------------------------------------ Phase 2 dataset (report numbers)
def dataset():
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    bars(ax[0], ["AT-6", "CS100", "CS-12", "AT-60"], [35.4, 31.0, 29.5, 4.0], [NAVY] * 4, "{:.1f}%",
         "Acquisition device (folds 1–9)", "% of records")  # fmt: skip
    bars(ax[1], ["NORM", "MI", "STTC", "CD", "HYP"], [43.6, 25.1, 24.0, 22.5, 12.2], [TEAL] * 5, "{:.1f}%",
         "Diagnostic superclass (multi-label)")  # fmt: skip
    bars(ax[2], ["SR", "AFIB", "STACH", "SARRH", "SBRAD", "PACE"], [76.9, 6.9, 3.8, 3.5, 2.9, 1.4], [ORANGE] * 6,
         "{:.1f}%", "Rhythm statement")  # fmt: skip
    save(fig, "dataset.png")


# ------------------------------------------------------------------ Phase 0 feasibility
def feasibility():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(3)
    lab = ["Flat line\n(history median)", "Repeat last 4 beats\n(sn4)", "Oracle: history beats\nat TRUE beat times"]
    ax.bar(x - 0.18, [0.110, 0.118, 0.074], 0.36, color=NAVY, label="MAE")
    ax.bar(x + 0.18, [0.110, 0.074, 0.058], 0.36, color=TEAL, label="DTW-MAE (timing-tolerant)")
    for i, (a, b) in enumerate([(0.110, 0.110), (0.118, 0.074), (0.074, 0.058)]):
        ax.text(i - 0.18, a, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.18, b, f"{b:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, lab, fontsize=9)
    ax.set(ylabel="mV (19,489 records)", title="Shape is predictable; timing is the bottleneck", ylim=(0, 0.135))
    ax.legend(frameon=False, fontsize=9)
    save(fig, "feasibility.png")


# ------------------------------------------------------------------ Phase 5 preprocessing (CNN tier, report)
def preprocessing():
    rows = [("Decomposition 0.67 Hz (input)", -0.34, -0.67, -0.12, True),
            ("8 independent leads", -0.2, -0.6, 0.1, False),
            ("Global standard scaling", 0.2, -0.3, 0.8, False),
            ("Gaussian noise aug. (replication)", 0.04, -0.06, 0.12, False),
            ("Time warping aug.", -0.48, -0.99, -0.16, False),
            ("Magnitude scaling aug.", -0.11, -0.23, -0.01, False),
            ("Baseline-wander aug.", 0.9, 0.7, 1.1, False),
            ("Mixup aug.", 0.0, -0.5, 0.3, False),
            ("Lead dropout aug.", 0.1, -0.5, 0.4, False)]  # fmt: skip
    notes = ["adopted (replicated)", "MAE n.s.", "DTW worse", "failed replication", "DTW +1.4 µV worse",
             "DTW worse", "worse", "n.s.", "n.s."]  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (lab, d, lo, hi, ok) in enumerate(rows):
        c = RED if ok else GREY
        ax.plot([lo, hi], [i, i], color=c, lw=2)
        ax.plot(d, i, "o", color=c, ms=7)
        ax.text(1.35, i, notes[i], va="center", fontsize=9, color=c)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-1.2, 2.3)
    ax.set(xlabel="ΔMAE vs default CNN (µV), 95% CI   ← better", title="Preprocessing effects are sub-µV (MAE ≈ 95 µV)")
    save(fig, "preprocessing.png")


# ------------------------------------------------------------------ Phase 7 data efficiency (report)
def learning_curve():
    frac = [5, 10, 20, 40, 60, 80, 100]
    mae = [0.1029, 0.0996, 0.0980, 0.0965, 0.0960, 0.0956, 0.0955]
    timing = [256, 259, 258, 259, 259, 257, 259]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(frac, mae, "o-", color=NAVY)
    ax[0].axhline(0.1161, color=BLACK, ls="--", lw=1)
    ax[0].text(50, 0.1150, "flat line 0.116", fontsize=9)
    ax[0].axvspan(60, 100, color=TEAL, alpha=0.12)
    ax[0].text(62, 0.1035, "≤ 0.6% from\nfull-data MAE", fontsize=9, color=TEAL)
    ax[0].set(xlabel="% of training patients (CNN, fold 9)", ylabel="MAE (mV)", title="MAE saturates")
    ax[1].plot(frac, timing, "o-", color=RED, label="CNN")
    ax[1].axhline(89, color=GREEN, ls=":", lw=1.5, label="sn4 (89 ms)")
    ax[1].axhline(298, color=BLACK, ls="--", lw=1, label="flat (298 ms)")
    ax[1].set(xlabel="% of training patients", ylabel="beat-timing error (ms)", title="More data does not fix rhythm",
              ylim=(0, 320))  # fmt: skip
    ax[1].legend(frameon=False, fontsize=9)
    save(fig, "learning_curve.png")


# ------------------------------------------------------------------ Phase 8 reliability (report)
def reliability():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars(ax, ["Patient bootstrap,\n3-seed ensembles\n(common practice)", "Run-and-patient\nbootstrap, 6 runs/arm\n(adopted)"],
         [41, 2.8], [GREY, RED], "{:.1f}%", "False 'wins' between identical models", "% of comparisons", (0, 50))  # fmt: skip
    ax.axhline(5, color="k", ls=":", lw=1)
    ax.text(1.35, 6, "nominal 5%", fontsize=8)
    save(fig, "reliability.png")


# ------------------------------------------------------------------ Phase 9 families (report)
FAM = pd.DataFrame({
    "family": ["CNN", "LSTM", "CNN-LSTM", "PatchTST*", "Transformer", "flat", "sn4"],
    "params_M": [1.85, 0.25, 0.72, 0.74, 1.18, 0, 0],
    "mae": [0.0961, 0.1030, 0.0911, 0.0963, 0.0944, 0.1161, 0.1218],
    "dtw": [0.0900, 0.0978, 0.0781, 0.0844, 0.0911, 0.1161, 0.0796],
    "template": [0.1050, 0.1045, 0.1288, 0.1160, 0.1090, 0.1048, 0.0642],
    "timing": [259, 254, 267, 276, 283, 298, 89],
    "min_per_run": [8, 33, 53, 160, 110, 0, 0],
})  # fmt: skip


def families(highlight=None, name="families.png"):
    cols = [NAVY] * 5 + [BLACK, GREEN]
    if highlight:
        cols = [RED if f == highlight else "#B8C2CC" for f in FAM.family[:5]] + [BLACK, GREEN]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    bars(ax[0], FAM.family, FAM.mae, cols, "{:.4f}", "MAE (mV) ↓", ylim=(0.08, 0.126))
    bars(ax[1], FAM.family, FAM.template, cols, "{:.3f}", "Template MAE (mV) ↓  [morphology]", ylim=(0, 0.145))
    bars(ax[2], FAM.family, FAM.timing, cols, "{:.0f}", "Beat timing (ms) ↓  [rhythm]", ylim=(0, 320))
    for a in ax:
        a.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Fold 9, single-model mean of 6 seeds, L1 objective", fontsize=10, y=1.02)
    save(fig, name)


# ------------------------------------------------------------------ Phase 10 objectives (report, fold 9)
def objectives():
    lab = ["L1", "L1 +\nderivative", "L1 +\nSTFT", "sn4-gated\nresidual", "flat", "sn4"]
    mae = [0.0911, 0.0915, 0.0935, 0.0890, 0.1156, 0.1218]
    tmpl = [0.1288, 0.1328, 0.0913, 0.0934, 0.1048, 0.0642]
    tim = [267, 268, 112, 125, 298, 89]
    cols = [GREY, GREY, TEAL, RED, BLACK, GREEN]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    bars(ax[0], lab, mae, cols, "{:.4f}", "MAE (mV) ↓", ylim=(0.08, 0.128))
    bars(ax[1], lab, tmpl, cols, "{:.3f}", "Template MAE (mV) ↓", ylim=(0, 0.15))
    bars(ax[2], lab, tim, cols, "{:.0f}", "Beat timing (ms) ↓", ylim=(0, 320))
    for a in ax:
        a.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("CNN-LSTM, fold 9, single-model mean of 6 seeds", fontsize=10, y=1.02)
    save(fig, "objectives.png")


# ------------------------------------------------------------------ OOF recomputation for the final config
def oof():
    import phase8_validation as p8
    import phase9_models as p9
    import phase10_objectives as p10

    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset

    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    y = sig[:, H : H + F].astype(np.float32)
    preds = {"final": np.zeros_like(y), "L1 CNN-LSTM": np.zeros_like(y)}
    for k in range(1, 10):
        rows = np.flatnonzero(meta.strat_fold == k)
        preds["final"][rows] = np.mean([np.load(p10.run_path("sn4_gated", 0.0, k, s))["pred"] for s in p8.CV_SEEDS], 0)
        preds["L1 CNN-LSTM"][rows] = np.mean([np.load(p9.run_path("cnn_lstm", 1e-3, False, k, s))["pred"] for s in p8.CV_SEEDS], 0)
    preds["sn4"] = np.load(p10.SN4)
    preds["flat"] = np.broadcast_to(np.median(sig[:, :H], axis=1, keepdims=True), y.shape).astype(np.float32)
    pid = meta.patient_id.to_numpy()
    acf = np.load(".cache/data/phase12/acf_peak.npy")

    def pm(v):
        return float(pd.Series(v).groupby(pid).mean().mean())

    lead = {a: [pm(np.abs(p[:, :, j] - y[:, :, j]).mean(1)) for j in range(12)] for a, p in preds.items()}
    err = {a: np.abs(p - y).mean(axis=(1, 2)) for a, p in preds.items()}
    q = pd.qcut(acf, 5, labels=False)
    per = {a: [pm(np.where(q == i, e, np.nan)) for i in range(5)] for a, e in err.items()}
    # lead chart
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = np.arange(12)
    for i, (a, c) in enumerate((("final", RED), ("L1 CNN-LSTM", GREY), ("sn4", GREEN), ("flat", BLACK))):
        ax.bar(x + (i - 1.5) * 0.2, lead[a], 0.2, color=c, label=a)
    ax.set_xticks(x, LEADS)
    ax.set(ylabel="patient-macro MAE (mV)", title="Error by lead (OOF, folds 1–9; final configuration)")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    save(fig, "leads.png")
    # periodicity chart
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for a, c in (("final", RED), ("L1 CNN-LSTM", GREY), ("sn4", GREEN), ("flat", BLACK)):
        ax.plot(range(5), per[a], "o-", color=c, label=a)
    ax.set_xticks(range(5), ["Q1\nleast", "Q2", "Q3", "Q4", "Q5\nmost"])
    ax.set(xlabel="history periodicity (ACF-peak quintile)", ylabel="MAE (mV)", title="The gain comes from periodic rhythms")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "periodicity.png")
    # success vs failure examples (fold 9)
    rows9 = np.flatnonzero(meta.strat_fold == 9)
    af = meta.rhythm_codes.apply(lambda c: bool(c & {"AFIB", "AFLT"})).to_numpy()
    e = err["final"]
    good_pool = rows9[(acf[rows9] > np.quantile(acf, 0.8)) & (e[rows9] < np.quantile(e[rows9], 0.25))]
    bad_pool = rows9[af[rows9] & (e[rows9] > np.quantile(e[rows9], 0.75))]
    rng = np.random.default_rng(0)
    good, bad = rng.choice(good_pool), rng.choice(bad_pool)
    fig, ax = plt.subplots(2, 1, figsize=(13, 5.4), sharex=True)
    t = np.arange(-200, F) / 100
    for a, r, ttl in ((ax[0], good, "Success: regular sinus rhythm (top-20% periodicity)"),
                      (ax[1], bad, "Failure: atrial fibrillation (irregular RR)")):  # fmt: skip
        a.plot(t[:200], sig[r, H - 200 : H, 1], color="0.55", lw=1, label="history")
        a.plot(t[200:], y[r, :, 1], color="k", lw=1.2, label="truth")
        a.plot(t[200:], preds["sn4"][r, :, 1], color=GREEN, lw=0.9, ls=":", label="sn4")
        a.plot(t[200:], preds["L1 CNN-LSTM"][r, :, 1], color=GREY, lw=1.1, label="L1 CNN-LSTM")
        a.plot(t[200:], preds["final"][r, :, 1], color=RED, lw=1.5, label="final")
        a.axvline(0, color="0.6", ls="--", lw=0.8)
        a.set(ylabel="mV (lead II)", title=f"{ttl} — final MAE {e[r]:.3f} mV")
    ax[0].legend(frameon=False, ncol=5, fontsize=9)
    ax[1].set_xlabel("time (s, 0 = forecast start)")
    save(fig, "examples.png")
    json.dump({"lead_mae": lead, "periodicity_mae": per, "examples_ecg_id": [int(meta.index[good]), int(meta.index[bad])]},
              (OUT / "oof_final.json").open("w"), indent=2)  # fmt: skip


# ------------------------------------------------------------------ compute (measured) + test
def compute():
    c = pd.DataFrame(json.load(open("reports/presentation/compute.json")))
    name = {"cnn": "CNN", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM", "patchtst": "PatchTST*", "transformer": "Transformer",
            "sn4_gated": "sn4-gated\n(final)"}  # fmt: skip
    c["label"] = c.model.map(name)
    train_min = {"cnn": 8, "lstm": 33, "cnn_lstm": 53, "patchtst": 160, "transformer": 110, "sn4_gated": 43}
    c["train_min"] = c.model.map(train_min)
    cols = [NAVY] * 5 + [RED]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    bars(ax[0], c.label, c.params / 1e6, cols, "{:.2f}", "Parameters (M)")
    bars(ax[1], c.label, c.train_min, cols, "{:.0f}", "Training time per run (min)†")
    bars(ax[2], c.label, c.peak_rss_train_MB, cols, "{:.0f}", "Peak RAM, training step (MB)")
    bars(ax[3], c.label, c.inference_ms_per_record, cols, "{:.2f}", "Inference (ms / record)‡")
    for a in ax:
        a.tick_params(axis="x", labelsize=8, rotation=0)
    save(fig, "compute.png")


def test():
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    lab = ["final", "L1\nCNN-LSTM", "flat", "sn4"]
    cols = [RED, GREY, BLACK, GREEN]
    bars(ax[0], lab, [0.0850, 0.0861, 0.1207, 0.1239], cols, "{:.4f}", "MAE (mV) ↓", ylim=(0.07, 0.13))
    bars(ax[1], lab, [0.0902, 0.1363, 0.1055, 0.0660], cols, "{:.3f}", "Template MAE (mV) ↓", ylim=(0, 0.15))
    bars(ax[2], lab, [110, 270, 299, 88], cols, "{:.0f}", "Beat timing (ms) ↓", ylim=(0, 330))
    fig.suptitle("Fold 10 (sealed until the end; 2,198 records, 1,904 unseen patients), 6-run ensembles", fontsize=10, y=1.02)
    save(fig, "test.png")


def factors():
    t = pd.DataFrame({"term": ["AF/flutter", "Hypertrophy (HYP)", "Paced", "Baseline wander (per e-fold)", "HR > 100",
                               "Age < 40", "Conduction (CD)", "Female", "CS-12 vs CS100", "HR < 60"],
                      "pct": [26.7, 23.7, 23.1, 23.0, 14.2, 14.2, 11.5, -6.3, -8.4, -11.1]})  # fmt: skip
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.barh(t.term, t.pct, color=[RED if v > 0 else TEAL for v in t.pct])
    for i, v in enumerate(t.pct):
        ax.text(v + (0.6 if v > 0 else -0.6), i, f"{v:+.0f}%", va="center", ha="left" if v > 0 else "right", fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-18, 34)
    ax.set(xlabel="adjusted effect on MAE (%), all CIs exclude 0", title="What makes a record hard (joint regression)")
    save(fig, "factors.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (dataset, feasibility, preprocessing, learning_curve, reliability, families, objectives, compute, test, factors):
        f()
    for fam in ("LSTM", "CNN-LSTM", "PatchTST*", "Transformer"):
        families(fam, f"families_{fam.strip('*').lower().replace('-', '_')}.png")
    oof()
    print("ok")
