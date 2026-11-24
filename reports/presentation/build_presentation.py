"""Build the study presentation (32 slides with speaker notes).

uv run --locked python reports/presentation/charts.py     # figures (needs caches)
uv run --locked python reports/presentation/explain.py    # saliency figures (needs caches)
uv run --locked python reports/presentation/build_presentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from pptx_writer import Deck  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports/presentation/figures"
REP = ROOT / "reports"
OUT = ROOT / "reports/presentation/PTBXL_ECG_forecasting.pptx"
NAVY, RED, TEAL, GREY, DARK, LIGHT = "1F2A44", "C0392B", "16A085", "7F8C8D", "222222", "EEF2F6"
W = 13.333

deck = Deck("Forecasting the 12-lead ECG on PTB-XL", "xm476")


def slide(section: str, title: str, source: str = ""):
    s = deck.add()
    s.shape(0, 0, 0.16, 7.5, fill=NAVY)
    s.text(
        0.45,
        0.22,
        10,
        0.35,
        [section.upper()],
        size=11,
        color=TEAL,
        bullet=False,
        bold=True,
        autofit=False,
    )
    s.text(0.45, 0.5, 12.4, 0.95, [title], size=25, color=NAVY, bullet=False, bold=True, anchor="t")
    s.shape(0.45, 1.42, 1.3, 0.05, fill=RED)
    if source:
        s.text(0.45, 7.05, 11.2, 0.32, [source], size=9, color=GREY, bullet=False, autofit=False)
    s.text(
        12.35,
        7.05,
        0.7,
        0.32,
        [str(len(deck.slides))],
        size=10,
        color=GREY,
        bullet=False,
        align="r",
        autofit=False,
    )
    return s


def key(s, x, y, w, h, text, size=14):
    s.text(
        x,
        y,
        w,
        h,
        [text],
        size=size,
        color=NAVY,
        bullet=False,
        fill=LIGHT,
        bold=True,
        anchor="ctr",
        inset=0.14,
    )


def notes(s, purpose, interp, discuss, beyond, qa):
    lines = ["PURPOSE", purpose, "", "KEY INTERPRETATION", *interp, "", "DISCUSSION POINTS", *discuss, "",
             "NOT ON THE SLIDE", *beyond, "", "LIKELY QUESTIONS"]  # fmt: skip
    for q, a in qa:
        lines += [f"Q: {q}", f"A: {a}"]
    s.notes = "\n".join(lines)


# =============================================================== 1 Title
s = deck.add()
s.shape(0, 0, W, 7.5, fill=NAVY)
s.shape(0.8, 3.55, 1.6, 0.06, fill=RED)
s.text(0.8, 1.3, 11.8, 2.2, ["Forecasting the 12-lead ECG on PTB-XL",
                             "What can be predicted, and how to measure it"], size=36, color="FFFFFF", bullet=False, bold=True)  # fmt: skip
s.text(0.8, 3.8, 11.8, 2.0, [
    "**Dataset:** PTB-XL v1.0.3 · 21,799 ten-second 12-lead ECGs · 18,869 patients · 100 Hz",
    "**Objective:** forecast the next 2.9 s of all 12 leads from a 7 s history — leakage-controlled, pre-registered, patient-level",
    "**Author:** xm476   ·   November 2026",
], size=16, color="DDE3EA", bullet=False)  # fmt: skip
s.text(0.8, 6.6, 11.8, 0.5, ["Phases 0–19 · decision rules fixed before each phase · sealed test fold opened once · every decision logged"], size=12,
       color="9FB3C8", bullet=False)  # fmt: skip
notes(
    s,
    "Introduce the study, the dataset and the deliverable: a leakage-controlled answer to how forecastable the 12-lead ECG is, and which modelling choices actually matter.",
    [
        "The study is organised as 20 phases (0-19), each with a pre-registration written before computation and a go/no-go decision.",
        "The headline: a CNN-LSTM that gates a causal 'repeat the last beats' forecast is the only model that beats every baseline-relevant criterion except the repeat-beats baseline's own morphology and timing; it held on the sealed test fold.",
    ],
    [
        "Emphasise the three design principles up front: pre-registration, patient-level statistics, and a test fold sealed in code until the final model was frozen.",
        "The forecast horizon is 2.9 s, not 3 s — explained on the next slide.",
    ],
    [
        "All training ran on CPU (40 cores, no GPU); this constrained model sizes (e.g. reduced PatchTST) and is disclosed throughout.",
        "The author is identified by the project username, xm476.",
    ],
    [
        (
            "Why PTB-XL?",
            "It is the largest public 12-lead ECG set with official patient-level stratified folds, demographic and device metadata, and a documented test fold — which makes leakage-free evaluation possible.",
        ),
        (
            "Is this a clinical tool?",
            "No. Everything here is signal-level evaluation on retrospective data; no clinical validation was performed.",
        ),
    ],
)

# =============================================================== 2 Problem
s = slide(
    "Problem statement",
    "Task: given 7 s of 12-lead ECG, forecast the next 2.9 s — in raw millivolts",
    "Sources: Phase 0, Phase 3 (D3.4) reports",
)
s.table(0.45, 1.75, [2.4, 4.0], [["Item", "Specification"],
                                 ["Data", "PTB-XL v1.0.3, official 100 Hz files"],
                                 ["Input", "7.0 s = 700 samples × 12 leads"],
                                 ["Output", "2.9 s = 290 samples × 12 leads"],
                                 ["Type", "Direct multi-step, deterministic"],
                                 ["Target", "Raw mV (no target scaling)"],
                                 ["Test fold", "Fold 10, sealed until the end"]],
        size=13, row_h=0.42)  # fmt: skip
s.text(7.2, 1.7, 5.7, 3.0, [
    "**Why 2.9 s, not 3 s?** The last 10 samples of every official 100 Hz file are a 500→100 Hz conversion artefact (up to 0.15 mV, verified against the 500 Hz originals) → trimmed",
    "**Why it matters**",
    ("Forecasting tests whether a model captures cardiac dynamics — rhythm and beat shape — not label shortcuts", 1),
    ("Motivations: gap filling and artefact repair, self-supervised pre-training objectives, early-warning research", 1),
    ("A hard, well-posed benchmark: a flat line is a strong baseline under pointwise error", 1),
], size=15)  # fmt: skip
key(s, 0.45, 5.4, 12.4, 1.2, "Core difficulty: beat morphology repeats, but beat timing drifts — and pointwise error punishes "
    "a well-shaped beat that is slightly late more than a flat line.")  # fmt: skip
notes(
    s,
    "Define the task precisely and pre-empt the '7 → 3 s' question.",
    [
        "The task as originally posed was 7 s → 3 s. Phase 3 found a conversion artefact in the last 100 ms of every 100 Hz file, so the target was shortened to 2.9 s (user decision D3.4). The 500 Hz originals were used only to diagnose the artefact; all modelling uses the official 100 Hz data.",
        "Forecasts are evaluated in raw mV so that numbers are physically interpretable (1 µV = 0.001 mV).",
    ],
    [
        "The difficulty statement at the bottom is the thread of the whole talk: it explains why MAE-trained models blur, and why the evaluation needed gatekeepers."
    ],
    [
        "A 7 s history contains on average 8-9 beats at the median heart rate (71 bpm), enough to estimate a beat template and a cycle length.",
        "The 500→100 Hz artefact reaches 0.15 mV, comparable to the size of model differences, so leaving it in would have contaminated the target.",
    ],
    [
        (
            "Why not use the 500 Hz data?",
            "The task specification is the 100 Hz release; 500 Hz would multiply compute 5× on CPU. At 100 Hz, QRS duration and QT cannot be resolved finely — a disclosed limitation.",
        ),
        (
            "Why direct multi-step instead of autoregressive?",
            "Direct forecasting avoids error accumulation and was the common protocol across all families; autoregressive decoding was not studied.",
        ),
    ],
)

# =============================================================== 3 Research questions
s = slide("Research questions", "Seven questions, each answered by a pre-registered phase")
s.table(0.45, 1.75, [0.7, 8.6, 3.1], [["#", "Question", "Phases"],
                                      ["RQ1", "How predictable is the next 2.9 s from 7 s, and what limits it?", "0"],
                                      ["RQ2", "Which data are trustworthy, and which splits and inputs are leakage-safe?", "1.5, 2, 3, 4"],
                                      ["RQ3", "How should forecasts be scored so that a flat line cannot win?", "6, 8"],
                                      ["RQ4", "Do preprocessing or more training data help?", "5, 7"],
                                      ["RQ5", "Which architecture is best under a fixed L1 objective?", "9"],
                                      ["RQ6", "Can a timing/shape-aware objective produce plausible, accurate beats?", "10, 11"],
                                      ["RQ7", "Does the final model hold across folds, subgroups and a sealed test fold?", "12–18"]],
        size=14, row_h=0.52, col_align=["ctr", "l", "ctr"])  # fmt: skip
key(
    s,
    0.45,
    6.15,
    12.4,
    0.65,
    "Every decision rule was fixed before the data were seen; deviations are disclosed as amendments.",
    13,
)
notes(
    s,
    "Give the audience the map of the talk: each question maps to phases and to slides.",
    [
        "RQ1-RQ4 are about the problem, the data and the evaluation; RQ5-RQ6 about modelling; RQ7 about whether conclusions survive stress tests."
    ],
    [
        "Point out that RQ3 (evaluation) changed the conclusions of RQ5: without gatekeepers the CNN-LSTM would have been declared the winner."
    ],
    [
        "Phase 1 (reproducible foundation) and Phase 19 (documentation) are infrastructure and not separate questions.",
        "Amendments (e.g. the 100-epoch cap in Phase 9, the hidden-size reversion in Phase 16) were recorded before any affected result was seen.",
    ],
    [
        (
            "What counts as 'pre-registered' here?",
            "A dated PREREGISTRATION.md per phase, written and (from Phase 10 on) committed before the runs, listing hypotheses, thresholds and decision rules.",
        )
    ],
)

# =============================================================== 4 Feasibility
s = slide(
    "RQ1 · Forecasting feasibility",
    "Beat shape is highly predictable; beat timing is the bottleneck",
    "Source: Phase 0 (non-learned baselines and oracles, 19,489 records, folds 1–9)",
)
s.image(FIG / "feasibility.png", 0.45, 1.7, 6.4, 4.2)
s.text(7.1, 1.7, 5.8, 3.9, [
    "**Perfect timing cuts MAE by ≈ 34%** (oracle 0.074 vs flat 0.110 mV)",
    "**Timing skill vanishes after ≈ 1 s**: knowing the true mean future heart rate helps 7%, knowing individual beat times 38%",
    "**Under MAE, a flat line beats repeating real beats** (0.110 vs 0.118) — but repeating wins on timing-tolerant DTW-MAE",
    "**Risk:** MAE/MSE-trained models are pushed towards blurred, flat forecasts",
], size=15)  # fmt: skip
key(
    s,
    0.45,
    6.05,
    12.4,
    0.8,
    "Decision (D0.8): keep 7 → 2.9 s, but evaluate with timing-aware metrics and require models to beat strong non-learned baselines.",
    13,
)
notes(
    s,
    "Establish what is achievable before any learning: the ceiling (oracle) and the floor (flat line, repeat-beats).",
    [
        "sn4 = average of the last four detected cycles, level-matched, repeated forward (computed from the history only). The oracle places history beats at the true future R-peak times: it is not achievable, but it bounds what perfect rhythm knowledge would give.",
        "The 34% gap between the flat line and the oracle is 'rhythm information'. A model can only close it by knowing when the next beats will occur.",
    ],
    [
        "The reversal between MAE and DTW-MAE for sn4 is the key insight: pointwise error punishes small timing errors more than it rewards correct shape.",
        "This is why the pre-registered feasibility rule technically returned 'no-go' for the task as specified, and why the task was reformulated rather than abandoned.",
    ],
    [
        "Error grows with lead time for all forecasts; sn4's advantage over the flat line in the first second disappears later (Phase 0 horizon figure).",
        "These baselines are deterministic, so they are cheap reference points reused in every later comparison.",
    ],
    [
        (
            "Why not shorten the horizon to 1 s?",
            "A 9 → 1 s variant was kept as a secondary analysis; the primary task was kept to preserve the original question, with evaluation adapted instead.",
        ),
        (
            "Is 34% an upper bound for learned models?",
            "Approximately: it is what perfect beat timing buys with history morphology. A model could in principle do better on morphology, but no model approached even the timing part.",
        ),
    ],
)

# =============================================================== 5 Dataset
s = slide(
    "Dataset overview",
    "19,571 clean development records from 16,945 patients; the acquisition device drives difficulty",
    "Source: Phase 2 report (folds 1–9; fold 10 metadata were not examined)",
)
s.image(FIG / "dataset.png", 0.45, 1.65, 12.4, 3.3)
s.table(0.45, 5.05, [2.5, 3.6, 2.5, 3.8], [
    ["Property", "Value", "Property", "Value"],
    ["Patients / records", "16,945 / 19,571 (cleaned)", "Repeat-patient records", "23% → patient-level statistics"],
    ["Age / sex", "mean 59.4 y (SD 16.7) / 52% male", "Independent leads", "8 of 12 (III, aVR, aVL, aVF derived)"],
    ["Heart rate", "median 71 bpm (IQR 63–83)", "Flat-line MAE by device", "0.080 (AT-60) – 0.131 mV (AT-6)"],
], size=11, row_h=0.42)  # fmt: skip
notes(
    s,
    "Describe the data the conclusions rest on, and the three dataset facts that shaped the design.",
    [
        "Fact 1: 23% of records come from patients with repeat recordings, so record-level statistics would be over-confident; every average is patient-macro and every CI is a patient-cluster bootstrap.",
        "Fact 2: only 8 of 12 leads are independent; limb leads III, aVR, aVL, aVF are exact linear combinations of I and II. This enables a lead-consistency plausibility check later.",
        "Fact 3: devices with little sub-0.5 Hz content (CS100, AT-60) are about twice as forecastable; device mix explains ~60% of the fold-9 difficulty gap.",
    ],
    [
        "Fold 9 is 100% human-validated and almost CS100-free (2.9% vs 34.6% in folds 1-8). It is harder by design, which is why it was validated against 9-fold CV before being used for model selection (Phase 8)."
    ],
    [
        "Height and weight are 68% / 57% missing and were not used; only age and sex are complete.",
        "Rhythm: 76.9% sinus; AF 6.9%; six rare rhythms have fewer than 150 records each, too few for reliable subgroup CIs.",
    ],
    [
        (
            "Were patient-conditioning inputs used?",
            "Metadata inputs were allowed by the leakage allow-list (age, sex, height, weight, device), but the final models use the signal only; conditioning was not a selected improvement in the phases run.",
        ),
        (
            "Is the device effect a confounder for model comparisons?",
            "Comparisons are paired on the same records, so device mix cancels; it matters for absolute numbers and subgroup reporting (slide 21).",
        ),
    ],
)

# =============================================================== 6 Data quality
s = slide(
    "Data quality",
    "Two cleaning decisions: exclude 30 defect records, trim a 100 ms conversion artefact",
    "Source: Phase 1.5 and Phase 3 reports (D3.1, D3.4)",
)
s.table(0.45, 1.75, [2.2, 4.3, 1.9], [
    ["Check", "Finding", "Decision"],
    ["Missing values", "0 missing samples; all signals finite, none saturated", "none needed"],
    ["Defect records", "30 (0.15%): 4 dropouts, 8 clipped plateaus, 18 limb-lead violations (61% paced)", "excluded (fixed rules)"],
    ["Conversion artefact", "Last 10 samples of every 100 Hz file: up to 0.15 mV vs 500 Hz originals", "trimmed → 2.9 s"],
    ["Duplicates", "0 exact, 0 near-duplicate pairs (r ≥ 0.99)", "none"],
    ["Noise", "Baseline wander drives difficulty (ρ = 0.63 with flat-line MAE); human quality flags are poor proxies", "kept; analysed as stratum"],
], size=11.5, row_h=0.6)  # fmt: skip
edge_figure = REP / "phase3_quality/fig6_edge_artifact.png"
with Image.open(edge_figure) as image:
    width, height = image.size
    image.crop((int(width * 0.52), 0, width, height)).save(FIG / "edge_artifact_zoom.png")
s.image(FIG / "edge_artifact_zoom.png", 8.85, 1.7, 4.1, 4.3)
key(
    s,
    0.45,
    5.6,
    8.2,
    1.1,
    "Cleaning removed 0.15% of records and 1% of each record; rules are applied identically to every fold, including fold 10.",
    13,
)
notes(
    s,
    "Show that the data were checked file by file, and justify the two cleaning decisions with evidence.",
    [
        "The artefact: comparing each 100 Hz file with its 500 Hz original shows that the final 100 ms deviate systematically (up to 0.15 mV), consistent with a filter edge effect in the official down-sampling. Keeping it would make the last 100 ms of every target partly artificial.",
        "Defect rules were fixed before inspection of outcomes and are applied identically to all folds, so they cannot bias model comparisons.",
    ],
    [
        "Noise is not 'cleaned' away: noisy records are part of the real distribution. Instead, baseline wander is used as a stratification variable in robustness and error analysis."
    ],
    [
        "43,604 files were SHA-256 verified at download; the upstream RECORDS index is malformed, so records are always indexed through ptbxl_database.csv.",
        "The 18 limb-lead violations are a clinically interesting subgroup (61% paced) but too few and too inconsistent to model.",
    ],
    [
        (
            "Could trimming have been avoided by using 500 Hz data?",
            "Yes, but at 5× compute; the 100 Hz task was specified, so trimming was the minimal, transparent fix.",
        ),
        (
            "Why not filter noise out?",
            "Phase 5 showed no filter preserves ECG morphology at the required level (slide 8); filtering the target would also change the task.",
        ),
    ],
)

# =============================================================== 7 Leakage
s = slide(
    "Data leakage prevention",
    "Patient-level official folds; fold 10 sealed in code; every input transform must be causal",
    "Source: Phase 4 and Phase 8 reports (D4.x, D8.2)",
)
s.table(0.45, 1.7, [3.3, 4.2, 2.6, 2.3], [
    ["Splitting method", "Evidence", "Leakage-safe?", "Used for"],
    ["Random record-level split", "≈ 20% of validation records share a patient with training; those score 3.2% better (memorisation)", "✗ leaks patients", "rejected"],
    ["Record-level K-fold", "same patient overlap", "✗", "rejected"],
    ["Official patient-level folds", "each patient in exactly one fold", "✓", "train 1–8, select on 9"],
    ["9-fold CV over folds 1–9", "same ranking as fold 9 (Kendall τ = 1.0)", "✓", "confirm finalists"],
    ["Nested CV", "72 trainings per config × seed", "✓ but redundant", "not used (fold 10 sealed)"],
], size=11.5, row_h=0.52)  # fmt: skip
s.table(0.45, 5.0, [3.3, 6.0, 3.1], [
    ["Other leakage channel", "Evidence", "Guard"],
    ["Labels / reports as inputs", "Extra-beat annotations 11.6× more frequent when the irregularity is only in the forecast window", "input allow-list"],
    ["Non-causal preprocessing", "Zero-phase filter predicts the future level 22% better than its causal twin", "automated causality test"],
], size=11, row_h=0.45)  # fmt: skip
notes(
    s,
    "Show which leakage channels exist in this task, how large they are, and how each was closed.",
    [
        "Forecasting has two leakage channels classification does not: (1) labels that describe the future part of the record, and (2) any preprocessing that looks across the history/forecast boundary (zero-phase filters, record-wide normalisation).",
        "All five channels now have code-level guards with tests: only patient-level splitters exist, fitted transforms record their training IDs, per-record transforms are tested for causality, and loading fold 10 raises an error unless explicitly enabled.",
    ],
    [
        "The memorisation probe quantifies patient leakage: records whose patient also appears in training score 3.2% better — a bias of the same order as the model differences being studied."
    ],
    [
        "Early stopping uses a separate 20% patient-level holdout inside the training folds, so the selection fold (9) is never used for stopping.",
        "Fold 10 was opened exactly once (Phase 17), after the final model and analysis were frozen with file hashes; a lock file prevents a second run.",
    ],
    [
        (
            "Why not stratify your own patient split?",
            "The official folds are already patient-level and label-stratified, and their use makes results comparable with the PTB-XL literature.",
        ),
        (
            "Does using fold 9 for selection bias the final estimate?",
            "Yes for fold 9 numbers; that is why finalists were confirmed with 9-fold CV and the final estimate comes from fold 10.",
        ),
    ],
)

# =============================================================== 8 Preprocessing
s = slide(
    "RQ4 · Preprocessing comparison",
    "Preprocessing barely matters: one lossless input transform adopted, everything else rejected",
    "Source: Phase 5 report (ridge tier: 20 variants; CNN tier: 3–6 seeds each; fold 9) and Phase 18 ablation",
)
s.image(FIG / "preprocessing.png", 0.45, 1.6, 7.3, 3.9)
s.table(8.0, 1.7, [1.6, 3.3], [
    ["Category", "Outcome"],
    ["Filters (7)", "all rejected: none preserves morphology (template r ≤ 0.980 < 0.99); 0.5 Hz high-pass distorts ST"],
    ["Scalers (6)", "none beats raw mV; per-record z-score +5.3 µV worse"],
    ["Features / PCA", "no significant gain"],
    ["Augmentations (7)", "none replicates; time warping lowers MAE but worsens DTW"],
    ["Adopted", "raw-mV target, 12 leads, causal 0.67 Hz baseline + residual input"],
], size=10.5, row_h=0.62, highlight=(5,))  # fmt: skip
key(s, 0.45, 5.75, 12.4, 1.0, "All effects are below 1 µV against an MAE of ≈ 95 µV. The adopted decomposition gained 0.34 µV (0.4%); "
    "for the final model, raw input was 1.1 µV better on fold 9 (not CV-confirmed) — the decomposition was kept.", 12.5)  # fmt: skip
notes(
    s,
    "Answer whether input engineering is a lever; it is not.",
    [
        "Two reference learners were used: ridge regression (all 20 variants, deterministic) and a 1.8 M-parameter CNN (shortlisted variants, multiple seeds).",
        "Filters were rejected on signal-level criteria first (morphology preservation and an impulse-displacement test) regardless of their effect on error: a filter that distorts the ST segment is unacceptable for an ECG forecast.",
        "The decomposition feeds the network a causal 0.67 Hz baseline estimate plus the residual; it is lossless and leaves the target unchanged. It passed the rule and a pre-specified replication with three new seeds.",
    ],
    [
        "Gaussian-noise augmentation passed once and failed replication — an early warning that sub-µV effects flip with training randomness, motivating the run-aware statistics on slide 19.",
        "Honesty point: the Phase 18 ablation on the final model found raw input better by 1.1 µV on fold 9. Because it was a single-fold result, the user decided to keep the decomposition (D18.1).",
    ],
    [
        "Seed-to-seed SD of the CNN's MAE is 0.2-0.3 µV, the same size as every preprocessing effect.",
        "8-lead input/output gives lead consistency by construction but did not improve MAE; for the CNN-LSTM, 8-lead output cost +2.9 µV (Phase 9).",
    ],
    [
        (
            "Why not a 0.05 Hz high-pass, as in diagnostic standards?",
            "The decomposition achieves the baseline separation losslessly; Butterworth high-passes at 0.5 Hz failed the impulse test, and a pure 0.05 Hz filter would barely change 7 s windows.",
        ),
        (
            "Would a Transformer respond differently to scaling?",
            "Possibly; this is disclosed. PatchTST uses its own per-instance normalisation (RevIN) internally.",
        ),
    ],
)

# =============================================================== 9 Evaluation
s = slide(
    "RQ3 · Evaluation framework",
    "Every pointwise metric ranks a 50 ms-late perfect ECG below a flat line → MAE + three gatekeepers",
    "Source: Phase 6 report (14 metrics scored against 6 axioms on controlled probe forecasts)",
)
s.table(0.45, 1.7, [2.2, 4.0], [
    ["Metric family", "Verdict"],
    ["MAE, RMSE, MSE, SNR, Pearson", "fail: reward blur and flat lines"],
    ["Beat F1", "fail: saturates beyond ±50 ms"],
    ["RR-interval error", "fail: undefined without beats"],
    ["DTW-MAE (±100 ms)", "pass: timing-tolerant shape"],
    ["Template MAE (median beat)", "pass: morphology"],
    ["Beat-timing error", "pass: most discriminative"],
], size=12, row_h=0.47)  # fmt: skip
s.table(6.95, 1.7, [1.7, 4.2], [
    ["Role", "Metric"],
    ["Primary", "MAE, 12 leads, patient-macro"],
    ["Gatekeeper 1", "DTW-MAE: shape allowing ±100 ms shift"],
    ["Gatekeeper 2", "Template MAE: median-beat morphology"],
    ["Gatekeeper 3", "Beat timing (ms): capped symmetric R-peak distance"],
    ["Reported", "≤ 50 ms share; R/ST/T amplitude errors"],
], size=12, row_h=0.47, highlight=(2, 3, 4))  # fmt: skip
key(s, 0.45, 5.25, 12.4, 1.35, "Decision rule (D6.2): A beats B only if the 95% CI of ΔMAE is entirely below 0 AND no gatekeeper's CI is entirely "
    "on the worse side. Baselines to beat: flat line (MAE) and repeat-beats sn4 (gatekeepers: 0.064 mV template, 89 ms timing).", 13)  # fmt: skip
notes(
    s,
    "Explain the evaluation that every later verdict depends on, and why it was needed.",
    [
        "Axioms were written first, e.g. A1: a perfect ECG shifted by 50 ms must score better than a flat line; A4: blurring must be penalised; A5: a missing beat must be penalised; A6: the metric must be defined for every forecast.",
        "Beat-timing error: R peaks are detected on lead II of [history | forecast]; each true beat is matched to the nearest predicted beat and vice versa; distances are capped at 300 ms and averaged; a forecast without beats scores the cap (300 ms).",
        "Template MAE compares the median beat of the forecast with the median beat of the truth, so it measures morphology independently of timing.",
    ],
    [
        "The rule is deliberately asymmetric: MAE must improve significantly, gatekeepers only must not get significantly worse. This blocks 'wins' bought by blurring.",
        "Interpretive bands for timing (D6.6): ≤ 25 ms ≈ detector noise; ≤ 50 ms acceptable; 50-150 ms misaligned; > 150 ms rhythm failure.",
    ],
    [
        "Beat timing was added post hoc in Phase 6 (disclosed) because beat F1 and RR error failed the axioms.",
        "Detector jitter on a perfect forecast with 0.05 mV noise is about 37 ms at the 90th percentile, so timing differences below ~25 ms are within measurement noise.",
        "A forecaster that knows the true future mean heart rate gets 57% of records within 50 ms — a reference target for 'beat-level timing skill'.",
    ],
    [
        (
            "Why not a single composite score?",
            "Composites hide trade-offs; a model could buy MAE with worse morphology. The gatekeeper rule makes trade-offs explicit.",
        ),
        (
            "Does the timing metric depend on the R-peak detector?",
            "Yes (NeuroKit2, lead II). It is applied identically to all models and baselines, but absolute values are detector-dependent — a disclosed limitation.",
        ),
    ],
)

# =============================================================== 10 Dataset size
s = slide(
    "RQ4 · Dataset size",
    "More data improves MAE with diminishing returns — and does nothing for rhythm",
    "Source: Phase 7 report (CNN reference, nested patient subsets, 3 seeds each, fold 9)",
)
s.image(FIG / "learning_curve.png", 0.45, 1.6, 8.3, 2.8)
s.table(0.45, 4.55, [1.2, 1.4, 1.3, 2.0, 1.4], [
    ["Patients", "Records", "MAE (mV)", "ΔMAE vs 100% (µV)", "Timing (ms)"],
    ["20%", "2,773", "0.0980", "+2.5 [+2.0, +3.0]", "258"],
    ["60%", "8,320", "0.0960", "+0.5 [+0.3, +0.7]", "259"],
    ["80%", "11,095", "0.0956", "+0.1 [−0.1, +0.3]", "257"],
    ["100%", "13,904", "0.0955", "–", "259"],
], size=11.5, row_h=0.4)  # fmt: skip
s.text(8.95, 1.7, 4.0, 4.9, [
    "**Minimum useful size** (beats flat): 20% — not robust (unseeded run: 80%)",
    "**Saturation** (within 1%, CI ∋ 0): 80% — not robust (unseeded: 60%)",
    "**60% of patients ≈ 0.6% from full-data MAE**",
    "Fitted MAE floor ≈ 0.094 mV (descriptive)",
    "**Timing ≈ 258 ms at every size**",
    "Decision: use the full pool (cost modest on CPU)",
], size=14)  # fmt: skip
notes(
    s,
    "Rule out 'more data' as the missing ingredient.",
    [
        "The learning curve flattens: going from 60% to 100% of patients gains about half a microvolt.",
        "Beat timing does not move at all from 5% to 100% of the data, which is the strongest evidence that the limitation is in the objective, not the sample size.",
    ],
    [
        "The exact 'minimum useful size' and 'saturation point' depended on the training run (seeded vs unseeded replicate gave different answers). Report them as ranges, not as precise thresholds.",
        "This phase also exposed a seeding defect: model weights were initialised before the training function set the seed. It was caught by a pre-registered bitwise check, fixed with a regression test, and all affected experiments were rerun.",
    ],
    [
        "Subsets are nested (each larger subset contains the smaller), so curves are monotone in data, not resampled.",
        "The power-law extrapolated floor (0.094 mV) is descriptive for this CNN; better architectures went below it (CNN-LSTM 0.087 in CV).",
    ],
    [
        (
            "Would a larger model need more data?",
            "Possibly; the curve was measured with the CNN reference. The CNN-LSTM was not re-profiled, a limitation.",
        ),
        (
            "Why keep 100% if 60% is almost as good?",
            "There is no cost to more data at this scale, and the curve is not perfectly flat at 80-100%.",
        ),
    ],
)

# =============================================================== 11 Roadmap
s = slide(
    "RQ5 · Model development roadmap",
    "Five families, one protocol: same input, same L1 loss, same stopping, same statistics",
    "Source: Phase 9 pre-registration and report; Phase 10 (objectives)",
)
fams = [("CNN\n(reference)", "local patterns"), ("LSTM", "recurrence over\n700 steps"), ("CNN-LSTM", "conv stem 700→175\n+ recurrence"),
        ("PatchTST*", "100 ms patch tokens,\nchannel-independent"), ("Transformer", "enc-dec, channel-\nmixing, 29 queries")]  # fmt: skip
for i, (n, d) in enumerate(fams):
    x = 0.45 + i * 2.55
    s.shape(x, 1.8, 2.15, 0.75, fill=NAVY, text=n, size=13)
    s.text(x, 2.6, 2.15, 0.8, [d], size=11, color=DARK, bullet=False, align="ctr")
    if i < 4:
        s.shape(x + 2.18, 2.0, 0.34, 0.35, prst="rightArrow", fill="B8C2CC")
for i, (n, d) in enumerate([("A · learning rate", "2 rates, internal holdout"), ("B · fold-9 screen", "6 seeds per family"),
                            ("C · 9-fold CV", "shortlist, 3 seeds/fold"), ("D · 8-lead output", "best family"),
                            ("Phase 10 · objectives", "top family")]):  # fmt: skip
    x = 0.45 + i * 2.55
    s.shape(x, 3.65, 2.15, 0.7, fill=TEAL if i < 4 else RED, text=f"{n}\n{d}", size=11)
    if i < 4:
        s.shape(x + 2.18, 3.83, 0.34, 0.35, prst="rightArrow", fill="B8C2CC")
s.text(0.45, 4.6, 12.4, 1.6, [
    "**Ordering logic:** increasing capacity for long-range temporal dependencies (local → recurrent → hybrid → attention), each a recognised forecasting baseline",
    "**Not a sequential search:** all families were trained in parallel under identical conditions, so differences are attributable to architecture",
    "Sized to similar CPU cost (0.25–1.85 M parameters). *PatchTST reduced from the published size (≈ 6 h/run on CPU)",
], size=14)  # fmt: skip
key(
    s,
    0.45,
    6.25,
    12.4,
    0.6,
    "Clean attribution: architecture first under a fixed loss; objectives changed only afterwards, on the best family.",
    13,
)
notes(
    s,
    "Explain how the model comparison was structured and why the families were chosen.",
    [
        "The requested 'LSTM → CNN-LSTM → PatchTST → Transformer' progression is the order of increasing modelling power for long sequences, but the experiment ran them side by side with a CNN reference carried from Phases 5-8. A sequential 'improve the last one' process was avoided because it confounds architecture with tuning effort.",
        "Holding the loss fixed (L1) was a deliberate scope choice (D9.0): the objective was already suspected to be the bottleneck, and varying both would make attribution impossible.",
    ],
    [
        "Stage A chose the learning rate per family on an internal holdout (not fold 9); Stage B screened on fold 9 with 6 seeds and the run-aware bootstrap; Stage C confirmed the shortlist with 9-fold CV (the user's multi-fold requirement, D8.6); Stage D tested 8-lead output."
    ],
    [
        "An amendment raised the epoch cap from 40 to 100 after Stage A, before any fold-9 result, because several families were still improving at epoch 40 (D9.1).",
        "Attention layers' private initialisation needed explicit seeding; this was fixed and tested before the runs.",
    ],
    [
        (
            "Why not larger Transformers?",
            "CPU budget: the published PatchTST size costs ~6 h per run; 6 seeds × 9 folds was infeasible. Results are 'per configuration', not 'per family at its best'.",
        ),
        (
            "Why include a CNN that is not in the requested list?",
            "It is the reference learner from Phases 5-8, linking architecture results to all earlier experiments.",
        ),
    ],
)

# =============================================================== 12 LSTM
s = slide(
    "RQ5 · LSTM",
    "LSTM: smallest and plausible, but the least accurate learned model",
    "Source: Phase 9 report; fold 9, 6 seeds; compare_runs 95% CIs",
)
s.image(FIG / "families_lstm.png", 0.45, 1.6, 12.4, 3.1)
s.table(0.45, 4.85, [2.6, 3.6], [
    ["Architecture", "1 layer, hidden 64, all 700 steps; 0.25 M params"],
    ["MAE single / ensemble", "0.1030 / 0.1021 mV"],
    ["vs CNN (ΔMAE)", "+6.8 µV [+4.3, +8.4] — worse"],
    ["Cost", "33 min/run; 0.23 s per training step"],
], size=11.5, row_h=0.42)  # fmt: skip
s.text(6.9, 4.8, 6.0, 2.1, [
    "**Strength:** beats flat on MAE and every gatekeeper; template MAE ≈ flat",
    "**Weakness:** recurrence over 700 raw steps is slow and small enough to under-fit",
    "**Failure mode:** forecasts level and average shape, no beats (timing 254 ms)",
    "Not shortlisted (worst MAE of the families)",
], size=14)  # fmt: skip
notes(
    s,
    "Summarise the LSTM result in context of the other families (highlighted in red on the chart).",
    [
        "The LSTM is the only family that passes all gatekeepers against the flat line, but only because it barely departs from a smooth mean forecast: its template MAE equals the flat line's.",
        "It is the least accurate learned model on MAE (+6.8 µV vs CNN).",
    ],
    [
        "The LSTM was sized to similar CPU cost as the others, which with 700 sequential steps means a small hidden size (64). That is a CPU-driven design choice, disclosed."
    ],
    [
        "A Week-5 LSTM (128×2) took about 1,000 s per epoch on CPU (Phase 5 decision D5.1), which is why the recurrent part was moved behind a convolutional stem in the CNN-LSTM.",
        "Seed SD of MAE was 0.2 µV, so the gap to other families is far larger than training noise.",
    ],
    [
        (
            "Would a bigger LSTM win?",
            "Possibly on MAE, but at prohibitive CPU cost; the CNN-LSTM achieves the same effect by shortening the sequence 4×.",
        ),
        (
            "Why is the LSTM's timing slightly better than the CNN-LSTM's?",
            "Both are near the flat-line regime (> 250 ms); differences of a few ms there are not meaningful.",
        ),
    ],
)

# =============================================================== 13 CNN-LSTM
s = slide(
    "RQ5 · CNN-LSTM",
    "CNN-LSTM: most accurate under L1 (−7.9% MAE vs CNN) but its beats are misshapen",
    "Source: Phase 9 report; fold 9 (6 seeds) and 9-fold CV (3 seeds × 9 folds)",
)
s.image(FIG / "families_cnn_lstm.png", 0.45, 1.6, 12.4, 3.1)
s.table(0.45, 4.85, [2.4, 1.9, 1.9], [
    ["vs LSTM / CNN", "fold 9", "9-fold CV"],
    ["ΔMAE (µV)", "−11.9 / −5.1", "– / −7.4 (9/9 folds)"],
    ["ΔTemplate (µV)", "+24 / +24", "– / +26 (p < 0.001)"],
    ["ΔTiming (ms)", "+13 / +7", "– / +12"],
], size=11.5, row_h=0.42)  # fmt: skip
s.text(6.9, 4.8, 6.0, 2.1, [
    "**Improvement:** tracks slow waves (T waves, baseline) of the next 1–2 beats in phase",
    "**New issue:** where beats appear they are misshapen → template worse than flat",
    "**Cost:** 0.72 M params, 53 min/run, 1.0 s/step (slowest per step); seeds disagree more (ensembling −3.7 µV)",
    "**Verdict:** no winner under the rule; carried to Phase 10",
], size=13.5)  # fmt: skip
notes(
    s,
    "Present the best L1 architecture and why it was still not declared the winner.",
    [
        "The CNN-LSTM's strided convolutional stem shortens 700 steps to 175, so a 2-layer LSTM (hidden 128) can model longer context cheaply. It is the most accurate family on MAE and DTW-MAE in every one of the 9 folds.",
        "But its template MAE is worse than the flat line in 9/9 folds: it produces beat-like structures that are mistimed or misshapen, which the morphology gatekeeper correctly penalises.",
        "Under the pre-registered rule, an MAE gain with a significantly worse gatekeeper is not a win.",
    ],
    [
        "This is the slide where the evaluation framework changes the conclusion. With MAE alone, CNN-LSTM is the clear winner (−7.9%).",
        "Its forecasts respect the limb-lead relations in 98% of records, and 8-lead output (deriving 4 leads) cost +2.9 µV MAE, so 12-lead output was kept.",
    ],
    [
        "Where the detector finds beats in CNN-LSTM forecasts (57% of records), the template MAE is 0.149 mV, against 0.102 mV where it finds none — misshapen beats are rated worse than no beats.",
        "R-amplitude error is slightly lower than the CNN's (0.842 vs 0.867 mV), but ST (0.082 vs 0.065) and T errors (0.286 vs 0.248) are higher.",
    ],
    [
        (
            "Is the MAE gain real or seed noise?",
            "Real: −7.4 µV with 9/9 folds better, p < 0.001; seed SD is 0.6 µV.",
        ),
        (
            "Then why not just pick it?",
            "Because it achieves accuracy by tracking slow waves while producing physiologically worse beats; the study's criterion requires both.",
        ),
    ],
)

# =============================================================== 14 PatchTST
s = slide(
    "RQ5 · PatchTST",
    "PatchTST (CPU-reduced): CNN-level accuracy at the highest cost",
    "Source: Phase 9 report; fold 9, 6 seeds; measured compute (slide 26)",
)
s.image(FIG / "families_patchtst.png", 0.45, 1.6, 12.4, 3.1)
s.table(0.45, 4.85, [2.6, 3.6], [
    ["Configuration", "100 ms patches, d 64, 2 layers, RevIN, channel-independent; 0.74 M"],
    ["vs LSTM", "MAE −6.7 µV (better)"],
    ["vs CNN-LSTM", "MAE +5.2 µV (worse); template better (0.116 vs 0.129)"],
    ["vs CNN", "MAE +0.2 µV (n.s.); DTW −5.5; template +11"],
], size=11, row_h=0.42)  # fmt: skip
s.text(6.9, 4.8, 6.0, 2.1, [
    "**Gain:** better DTW-MAE than CNN — some shape tracking",
    "**Limitation:** reduced from published size (patch 16/stride 8, d 128, 3 layers ≈ 6 h/run on CPU)",
    "**Cost:** 160 min/run, 937 MB peak RAM — most expensive family",
    "Not shortlisted; result applies to this configuration only",
], size=13.5)  # fmt: skip
notes(
    s,
    "Place PatchTST relative to LSTM and CNN-LSTM, and be explicit about the CPU-driven reduction.",
    [
        "Channel-independent patching treats each lead separately (with its baseline and residual channels), which removes cross-lead information that the other families can use.",
        "Its accuracy matches the CNN reference while costing 20× more training time.",
    ],
    [
        "The result should not be read as 'PatchTST is weak': the published configuration was infeasible on CPU for 6 seeds × multiple folds. With a GPU it should be re-run at full size."
    ],
    [
        "Patch length 100 ms (10 samples × the lead's 2 input channels) is about the duration of a QRS complex, so a single patch can hold a whole QRS — but the model still did not forecast QRS complexes.",
        "Its timing (276 ms) is in the flat-line regime like all L1 families.",
    ],
    [
        (
            "Why channel-independent?",
            "That is PatchTST's published design and its main inductive bias; the channel-mixing alternative is the Transformer on the next slide.",
        ),
        (
            "Would a GPU change the ranking?",
            "Unknown; it is the clearest case where compute limited the comparison, listed under future work.",
        ),
    ],
)

# =============================================================== 15 Transformer
s = slide(
    "RQ5 · Transformer",
    "Encoder-decoder Transformer: second-best MAE and generalises, but slow and unstable across seeds",
    "Source: Phase 9 report; fold 9 (6 seeds), 9-fold CV; measured compute",
)
s.image(FIG / "families_transformer.png", 0.45, 1.6, 12.4, 3.1)
s.table(0.45, 4.85, [2.5, 3.7], [
    ["Configuration", "70 tokens × 100 ms × 24 ch; d 128, 8 heads, 3+2 layers; 1.18 M"],
    ["9-fold CV MAE", "0.0916 (CNN-LSTM 0.0866, CNN 0.0940)"],
    ["vs CNN (CV)", "MAE −2.4 µV (9/9 folds); template +3.0 µV (p = 0.04)"],
    ["Generalisation", "fold-9 rank = CV rank for every pair"],
], size=11, row_h=0.42)  # fmt: skip
s.text(6.9, 4.8, 6.0, 2.1, [
    "**Accuracy:** 2nd of 5 on MAE; DTW no better than CNN",
    "**Instability:** largest seed spread (SD 1.2 µV); timing ranges from flat-like to beat-like between seeds",
    "**Cost:** 110 min/run, 829 MB RAM, 0.61 ms/record",
    "Carried to Phase 10 by the rule, then dropped for compute (D10.0)",
], size=13.5)  # fmt: skip
notes(
    s,
    "Complete the architecture comparison and state the Phase 9 conclusion: no family is best under L1.",
    [
        "The Transformer mixes channels and decodes 29 learned horizon queries, each producing 100 ms × 12 leads. It ranks second on MAE in CV, better than the CNN in all 9 folds, but with a small, significant morphology penalty.",
        "Phase 9 conclusion: every MAE gain over the CNN comes with a significantly worse morphology gatekeeper, so no family is declared best; none approaches sn4 on morphology or timing.",
    ],
    [
        "Deviation to disclose: the pre-registered rule carried CNN-LSTM and Transformer into Phase 10, but the user dropped the Transformer to halve compute (D10.0). The objective findings therefore apply to the CNN-LSTM only."
    ],
    [
        "Generalisation check: fold-9 verdicts agreed with 9-fold CV on every pair, validating fold-9 screening again (hypothesis H9.3).",
        "Seed instability means single-run Transformer results would be unreliable; the 6-run protocol was essential here.",
    ],
    [
        (
            "Could the Transformer benefit more from the sn4 gate than the CNN-LSTM?",
            "Untested; listed as future work.",
        ),
        (
            "Why is DTW not better despite lower MAE?",
            "Its gain is mostly in the level and slow trend, not in shape-tracking.",
        ),
    ],
)

# =============================================================== 16 Objectives
s = slide(
    "RQ6 · Training objectives",
    "Gating the repeat-beats forecast is the only change that improves accuracy AND plausibility",
    "Source: Phase 10 report; chart: fold 9, 6 seeds; table: 9-fold CV, 3-seed ensembles",
)
s.image(FIG / "objectives.png", 0.45, 1.6, 12.4, 3.1)
s.table(0.45, 4.85, [2.6, 1.35, 1.35, 1.35, 1.35, 1.6], [
    ["9-fold CV", "MAE", "DTW", "Template", "Timing", "≤ 50 ms"],
    ["L1 (Phase 9)", "0.0866", "0.0757", "0.131", "270 ms", "0.2%"],
    ["L1 + STFT (λ 0.3)", "0.0873", "0.0703", "0.092", "118 ms", "14%"],
    ["sn4-gated residual", "0.0850", "0.0676", "0.088", "109 ms", "36%"],
    ["sn4 baseline", "0.1167", "0.0735", "0.065", "88 ms", "44%"],
], size=11, row_h=0.36, highlight=(3,))  # fmt: skip
s.text(10.4, 4.8, 2.6, 2.2, [
    "**Derivative:** no effect (template +4 µV)",
    "**STFT:** shape ✓, MAE ✗ (+0.7 µV, 0/9 folds)",
    "**sn4-gated:** beats L1 and flat on all 4 metrics, 9/9 folds",
], size=12)  # fmt: skip
notes(
    s,
    "Present the key modelling result of the study.",
    [
        "sn4-gated residual model: forecast = sigmoid(gate(h)) ⊙ sn4 + correction(h), where h is the CNN-LSTM's summary of the history and sn4 is computed causally from the history. A closed gate recovers the plain CNN-LSTM; an open gate with zero correction recovers sn4.",
        "It beats both the flat line and the L1 CNN-LSTM on MAE and all three gatekeepers in 9 of 9 folds: MAE −1.6 µV, template −43 µV, timing −161 ms vs L1.",
    ],
    [
        "The STFT (multi-resolution spectral magnitude) loss shows the trade-off clearly: it fixes shape and timing (118 ms) but costs MAE, so it fails the MAE-primary rule. Larger λ gave timing near sn4 (78-80 ms) but +10-23% MAE.",
        "No objective beats sn4 itself on template MAE or timing — the gated model's beats are damped copies of sn4's.",
    ],
    [
        "λ for auxiliary losses was chosen on the internal holdout (never fold 9) from {0.3, 1, 3}; derivative λ = 0.3, spectral λ = 0.3 (fallback rule: no λ met the 2% MAE tolerance).",
        "The gated model adds a second linear head (+0.45 M parameters); capacity was not controlled, but Phase 18's ablation shows the gain comes from the sn4 skip path, not capacity (slide 25).",
        "Soft-DTW was deferred: 10-30× slower on CPU.",
    ],
    [
        (
            "Isn't using sn4 as input 'cheating'?",
            "No: sn4 is computed from the 7 s history only, exactly what any model sees. It is a hand-crafted causal feature with a skip connection to the output.",
        ),
        (
            "Why does the gate damp beats rather than copy them?",
            "Under L1, when beat timing is uncertain, a damped beat has lower expected error than a full one at a possibly wrong time. The ablation without a gate copies full beats and loses 16 µV MAE.",
        ),
    ],
)

# =============================================================== 17 Final model architecture
s = slide(
    "RQ6 · Final model architecture",
    "sn4-gated CNN-LSTM: the network learns how far to trust the repeated beats, step by step and lead by lead",
    "Sources: src/ptbxl_forecast/models/gated_baseline.py; Phase 0 (sn4), Phase 10 (objective), Phase 11 (settings), Phase 16 (frozen hidden-128 configuration)",
)
s.shape(
    0.45, 1.8, 2.0, 1.15, fill=GREY, text="7 s history\n700 × 24 ch\nbaseline + residual", size=12
)
s.shape(2.55, 2.2, 0.4, 0.35, prst="rightArrow", fill=GREY)
s.shape(
    3.05,
    1.8,
    2.0,
    1.15,
    fill=NAVY,
    text="Conv stem\n2 × Conv1d (k 7, s 2)\n700 → 175 steps",
    size=12,
)
s.shape(5.15, 2.2, 0.4, 0.35, prst="rightArrow", fill=GREY)
s.shape(5.65, 1.8, 1.7, 1.15, fill=NAVY, text="LSTM\n2 layers\nhidden 128 → h", size=12)
s.shape(7.45, 2.2, 0.4, 0.35, prst="rightArrow", fill=GREY)
s.shape(7.95, 1.65, 2.2, 0.65, fill=TEAL, text="Gate g = σ(Linear(h))", size=12)
s.shape(7.95, 2.45, 2.2, 0.65, fill=TEAL, text="Correction c = Linear(h)", size=12)
s.shape(1.25, 3.05, 0.35, 0.55, prst="downArrow", fill=GREY)
s.shape(
    0.45,
    3.7,
    4.6,
    1.0,
    fill=GREY,
    text="sn4 (no learning, causal)\nperiod from multi-lead ACF of the history;\nmean of last 4 cycles, level-matched, repeated",
    size=12,
)
s.shape(5.15, 4.03, 2.7, 0.35, prst="rightArrow", fill=GREY)
s.shape(8.9, 3.17, 0.35, 0.45, prst="downArrow", fill=GREY)
s.shape(7.95, 3.7, 2.2, 1.0, fill=RED, text="ŷ = g ⊙ sn4 + c\n290 × 12, raw mV", size=14)
s.text(10.4, 1.65, 2.6, 3.2, [
    "**Gate:** one value per horizon step × lead (3,480)",
    "**g → 0:** plain L1 CNN-LSTM",
    "**g → 1, c = 0:** sn4",
    "**Skip path:** sn4 bypasses the network, so it cannot be ignored",
], size=12)  # fmt: skip
s.table(0.45, 4.95, [2.6, 9.8], [
    ["Setting", "Value"],
    ["Training", "L1 loss, AdamW, lr 1e-3, batch 64, early stopping on a 20% internal patient holdout (≤ 100 epochs)"],
    ["Size / data", "1.17 M parameters (+0.45 M vs L1 CNN-LSTM: gate head); 15,592 training records from folds 1–9"],
    ["Final forecast", "Mean of 6 seeds; weights, pipelines and config frozen with SHA-256 hashes before fold 10 was opened"],
], size=11.5, row_h=0.36)  # fmt: skip
key(
    s,
    0.45,
    6.45,
    12.4,
    0.5,
    "The network never has to learn beat timing: sn4 supplies it causally; the network decides when to trust it.",
    12.5,
)
notes(
    s,
    "Make the final model concrete: the rest of the talk (robustness, plausibility, ablations, test) refers to its gate and skip path.",
    [
        "sn4 is the Phase 0 seasonal-naive baseline: the cycle length is estimated from the autocorrelation of the first difference of all leads (searched near the median R-R interval of the history), and the forecast is the average of the last four cycles, each level-matched to the most recent cycle. It uses the history only.",
        "The CNN-LSTM body is the Phase 9 model. Its last hidden state h feeds two linear heads: a sigmoid gate (290 steps × 12 leads) that scales sn4, and a correction that is added. This is a residual formulation around a strong causal prior.",
    ],
    [
        "Limiting cases make the design easy to reason about: a closed gate gives back the L1 CNN-LSTM, an open gate with no correction gives back sn4. The model can only do better than both by choosing between them per step and lead.",
        "The gate being per lead is also its weakness: limb-lead relations (III = II − I, etc.) are no longer guaranteed (Phase 13, slide 23).",
    ],
    [
        "Phase 11 tuned on fold 9 and preferred hidden 256; 9-fold CV reversed this (D12.1), so the frozen model uses hidden 128 (slide 18).",
        "sn4 costs ≈ 7.5 ms per record on CPU (R-peak detection + period estimation) and dominates inference time (slide 26).",
    ],
    [
        (
            "Why not feed sn4 as an input channel instead?",
            "Phase 18 tested exactly that: the network then ignores it (timing 264 ms, the same as without sn4). The skip path is essential (slide 25).",
        ),
        (
            "Why a sigmoid gate rather than a free weight?",
            "It keeps the model between the two interpretable limits (0 = no sn4, 1 = full sn4) and makes the gate readable as trust (slide 24).",
        ),
    ],
)

# =============================================================== 18 Tuning
s = slide(
    "RQ6 · Hyperparameter tuning",
    "Only a larger LSTM lowered MAE — and every setting that lowered MAE less forecast sharper, better-timed beats",
    "Source: Phase 11 report (stage A: internal holdout, seed 42, one change at a time; stage B: fold 9, 6 vs 6 runs); Phase 12 (9-fold CV); D12.1",
)
s.table(0.45, 1.7, [2.1, 1.0, 1.15, 1.15, 1.25, 1.15], [
    ["Stage A", "Params", "MAE", "DTW", "Template", "Timing"],
    ["base (Phase 10)", "1.17 M", "0.0865", "0.0691", "0.0912", "119"],
    ["hidden 256", "2.68 M", "0.0855", "0.0668", "0.0913", "124"],
    ["hidden 64", "0.56 M", "0.0905", "0.0743", "0.0867", "96"],
    ["3 LSTM layers", "1.30 M", "0.0872", "0.0699", "0.0986", "140"],
    ["stem width 128", "1.30 M", "0.0877", "0.0691", "0.0876", "113"],
    ["lr 3e-4", "1.17 M", "0.0919", "0.0733", "0.0851", "91"],
    ["+ spectral λ 0.1", "1.17 M", "0.0879", "0.0666", "0.0840", "109"],
    ["+ spectral λ 0.3", "1.17 M", "0.0902", "0.0630", "0.0745", "95"],
], size=11.5, row_h=0.42, highlight=(2,))  # fmt: skip
s.text(8.5, 1.65, 4.4, 3.9, [
    "**Shortlist rule:** ≥ 0.5 µV better holdout MAE → only hidden 256",
    "**Stage B (fold 9, 6 vs 6):** ΔMAE −0.77 µV [−1.44, −0.18]; gatekeepers n.s. → adopted",
    "**9-fold CV (Phase 12):** MAE −1.0 µV, but template +1.3 µV (8/9 folds worse) and timing +5.2 ms (9/9) → **reverted to hidden 128** before the test (D12.1)",
    "**Not tested:** interactions (e.g. hidden 256 + spectral), batch size, dropout",
], size=12.5)  # fmt: skip
key(s, 0.45, 5.75, 12.4, 1.05, "Trade-off: spectral λ 0.3 gave the best template (0.075 mV) and timing (95 ms) of any learned model at +3.6 µV MAE. "
    "The MAE-primary rule rejects it — a rule that weighted morphology would choose differently.", 12.5)  # fmt: skip
notes(
    s,
    "Show what was tuned, what was adopted, and why the adoption was later reversed.",
    [
        "Stage A changed one hyperparameter at a time from the Phase 10 configuration and trained one seed per candidate on the internal patient holdout (fold 9 untouched). Only hidden 256 met the pre-registered shortlist threshold.",
        "Stage B confirmed hidden 256 on fold 9 with 6 runs per arm: a small but significant MAE gain (0.9%) at 2.3× the parameters and about 2× the training time.",
        "Phase 12's 9-fold CV then showed a consistent morphology and timing cost, so the user reverted the final model to hidden 128 while fold 10 was still sealed.",
    ],
    [
        "The grid shows the central tension of the study in miniature: every change that makes the model less MAE-optimal (smaller, slower learning rate, spectral term) produces sharper, better-timed beats.",
        "Stage A is single-seed and descriptive; only the shortlisted candidate was tested with the full statistics.",
    ],
    [
        "Selection criterion and its alternative are both pre-registered in spirit: MAE is primary, gatekeepers must not be worse (D6.2).",
        "Stage A timings: hidden 256 took 88 min vs 45 min for base on CPU.",
    ],
    [
        (
            "Isn't the hidden-size reversal a form of test-set peeking?",
            "No: it used only folds 1–9 (cross-validation). Fold 10 was opened afterwards, once.",
        ),
        (
            "Why not adopt spectral λ 0.3?",
            "It fails the MAE-primary rule (+3.6 µV on the holdout). It is listed as future work in combination with the gate.",
        ),
    ],
)

# =============================================================== 19 Reliability
s = slide(
    "Statistical reliability",
    "Training randomness decides sub-µV verdicts → 6 runs per arm, run-and-patient bootstrap",
    "Source: Phase 8 (calibration on 18 seeded runs per configuration), Phases 11, 12, 17",
)
s.image(FIG / "reliability.png", 0.45, 1.6, 5.4, 3.5)
s.table(6.1, 1.7, [2.5, 4.3], [
    ["Practice", "Implementation"],
    ["Multiple seeds", "6 independent seeds per arm (fold 9); 3 seeds × 9 folds (CV); final = 6-seed ensemble"],
    ["Bootstrap", "resample training runs, then patients (23% repeat patients); 1,000 draws"],
    ["Confidence intervals", "95% percentile CI on MAE and all 3 gatekeepers"],
    ["Significance", "D6.2 rule on CIs; paired t-tests across 9 folds (approximate)"],
    ["Reproducibility", "bitwise-identical reruns verified at every phase"],
], size=11, row_h=0.55)  # fmt: skip
key(s, 0.45, 5.35, 12.4, 1.45, "Example — why multi-fold confirmation matters: hidden 256 beat hidden 128 on fold 9 (ΔMAE −0.77 µV [−1.44, −0.18]), "
    "but 9-fold CV showed worse template (+1.3 µV, 8/9 folds) and timing (+5.2 ms, 9/9 folds) → final model reverted to hidden 128 before the test.", 12.5)  # fmt: skip
notes(
    s,
    "Show that reported differences are not artefacts of training randomness or patient clustering.",
    [
        "Calibration: 18 independently seeded runs per configuration allowed comparing identical configurations against each other. The common practice (patient bootstrap of 3-seed ensembles) declared a 'win' in 41% of such null comparisons. Resampling runs as well as patients, with 6 runs per arm, brought this to 2.8% while keeping 100% power for a −1.8 µV effect.",
        "These rates are empirical for this setting (correlated draws), not universal guarantees.",
    ],
    [
        "The hidden-size example is the best illustration in the study: a pre-registered fold-9 rule selected hidden 256; the subsequent cross-validation revealed a consistent morphology/timing cost, and the user reverted before opening the test fold."
    ],
    [
        "9-fold CV p-values are approximate because training sets overlap between folds (each pair shares 7 of 8 training folds), and seed noise enters the between-fold variance.",
        "Every phase included a bitwise rerun check of a cached run; one such check exposed the Phase 7 seeding defect.",
    ],
    [
        (
            "Why 6 runs?",
            "It is the smallest number that brought the calibrated false-positive rate below 5% in Phase 8 at acceptable CPU cost.",
        ),
        (
            "Was multiplicity corrected?",
            "No family-wise correction; instead decisions were limited to pre-registered comparisons and confirmed across folds, and post hoc analyses are labelled.",
        ),
    ],
)

# =============================================================== 20 Robustness
s = slide(
    "RQ7 · Robustness",
    "Stable across folds, devices, quality strata and noise; it breaks for AF/flutter and a missing lead II",
    "Source: Phase 12 report — strata: OOF folds 1–9, 3-seed ensembles, patient-cluster bootstrap; perturbations: fold 9, 6 models. Analysed the hidden-256 variant (≈ 1 µV MAE from final)",
)
s.table(0.45, 1.65, [2.6, 1.05, 1.05, 1.05, 1.05, 1.05, 1.3], [
    ["Stratum (patients)", "Final MAE", "L1 MAE", "sn4 MAE", "Final timing", "sn4 timing", "Beats flat / L1 / sn4"],
    ["AT-6 device (6,579)", "0.0923", "0.0958", "0.1343", "106", "82", "✓ / ✓ / ✗"],
    ["CS100 device (4,712)", "0.0796", "0.0825", "0.0969", "116", "93", "✓ / ✓ / ✗"],
    ["High baseline wander (6,177)", "0.1008", "0.1047", "0.1478", "116", "91", "✓ / ✓ / ✗"],
    ["Sinus rhythm (13,517)", "0.0800", "0.0816", "0.1119", "106", "80", "✓ / ✓ / ✗"],
    ["AF / flutter (1,143)", "0.1141", "0.1141", "0.1575", "197", "161", "✓ / ✗ / ✗"],
    ["Paced (206)", "0.1205", "0.1680", "0.1355", "102", "93", "✓ / ✓ / ✗"],
    ["HR > 100 bpm (1,379)", "0.1126", "0.1227", "0.1433", "96", "81", "✓ / ✓ / ✗"],
], size=11, row_h=0.42, highlight=(5,))  # fmt: skip
s.table(0.45, 5.15, [2.6, 1.3, 1.3, 1.3, 1.3], [
    ["History perturbation", "MAE", "Template", "Timing", "Beats flat"],
    ["clean / noise σ 0.05 mV", "0.0882 / 0.0889", "0.093 / 0.094", "126 / 128", "✓"],
    ["lead II set to 0", "0.0892", "0.118", "262", "✗ (template)"],
], size=11, row_h=0.4, highlight=(2,))  # fmt: skip
s.text(9.95, 1.6, 3.0, 5.3, [
    "**Folds:** OOF MAE 0.0840 mV, fold SD 0.0009; beats flat and L1 in 9/9 folds",
    "**H12.1** beats flat in every stratum ✓",
    "**H12.2** beats L1 in every stratum ✗ (AF/flutter: tie)",
    "**H12.3** AF timing ≤ sn4 ✗ (+36 ms)",
    "**H12.4** beats flat under every perturbation ✗ (lead II)",
    "Largest gains over L1: paced −47.5 µV, HR > 100 −10.0 µV",
], size=12)  # fmt: skip
notes(
    s,
    "Show where the final model holds and where it breaks, using the pre-registered robustness hypotheses.",
    [
        "Across folds the result is very stable (fold SD below 1 µV) and the model beats flat in every stratum, including every device and quality group.",
        "AF/flutter is the one stratum with no gain over L1: sn4 copies regular beats into an irregular rhythm, so the gate has little to trust, and timing is worse than sn4 itself.",
        "Gaussian noise in the history barely matters (+0.7 µV MAE at 0.05 mV). Zeroing lead II removes most timing skill, because lead II drives both sn4's period search and the metric's beat detection.",
    ],
    [
        "The largest gains are in regular, fast or paced rhythms where sn4 is informative — the mirror image of the AF failure.",
        "The lead-II result is partly a measurement artefact (the timing metric detects beats on lead II) and partly real (sn4 depends on lead II peaks).",
    ],
    [
        "Phase 12 also produced the 9-fold CV evidence that reversed the hidden-size choice (slide 18).",
        "Strata use one OOF ensemble per record; rhythm labels are PTB-XL statements, not adjudicated; perturbations are synthetic.",
    ],
    [
        (
            "Why does the device effect disappear here when it drove difficulty in Phase 2?",
            "Devices differ in absolute difficulty (the MAE column), but the model's advantage over flat and L1 holds within every device; Phase 14 shows the device effect vanishes after adjusting for patient factors.",
        ),
        (
            "Would a different lead for sn4 fix the lead-II failure?",
            "Likely partly; a multi-lead or lead-robust period estimate is a cheap follow-up, not tested here.",
        ),
    ],
)

# =============================================================== 21 Error analysis
s = slide(
    "Error analysis",
    "Errors concentrate in irregular rhythms, pacing, hypertrophy and baseline wander",
    "Sources: Phases 12, 14, 15 (OOF folds 1–9). Regression on the hidden-256 variant (±1 µV); periodicity recomputed for the final model",
)
s.image(FIG / "factors.png", 0.45, 1.6, 6.1, 3.7)
s.image(FIG / "periodicity.png", 6.75, 1.6, 6.1, 3.7)
s.text(0.45, 5.45, 12.4, 1.5, [
    "**Hardest groups:** AF/flutter (+27%), pacing (+23%), hypertrophy (+24%), baseline wander (+23% per e-fold), HR > 100 (+14%); device effect vanishes after adjustment (AT-6 vs CS100 +0.9%, n.s.)",
    "**Worst 1% of records:** high baseline wander 72% (vs 33%), AF 26% (vs 7%), paced 13% (vs 1%)",
    "**Gain over L1 lives in periodic histories:** −14.0 µV in the most periodic quintile, +2.4 µV (worse) in the least periodic",
], size=13)  # fmt: skip
notes(
    s,
    "Identify typical failure patterns and who the model works for.",
    [
        "Joint regression of log MAE on age, sex, device, rhythm, heart rate, diagnostic superclasses and baseline wander, with patient-cluster bootstrap CIs. Effects are associations, not causes.",
        "The periodicity curve (ACF-peak quintiles of the history) is the clearest mechanism: when the history is regular, the gate trusts sn4 and error drops; when it is irregular, sn4 is misleading and the model is slightly worse than the plain L1 CNN-LSTM.",
    ],
    [
        "The unadjusted device effect seen in Phase 2 is mostly baseline wander and case mix — once wander is in the model, AT-6 is no worse than CS100.",
        "AF/flutter is the main subgroup where the final model gives no benefit over L1 (ΔMAE +0.0 µV) and its timing is worse than sn4 (197 vs 161 ms).",
    ],
    [
        "Typical failure in AF: forecast beats copied from sn4 at regular intervals while true beats are irregular; or, with the gate closed, a flat slow-wave forecast (example on slide 23).",
        "Paced rhythms benefit most from the gate (−47.5 µV vs L1): pacing makes the rhythm highly regular.",
    ],
    [
        (
            "Should AF be handled by a separate model?",
            "That, or a periodicity-aware gate/fallback, is the most direct improvement suggested by this analysis (future work).",
        ),
        (
            "Why is age < 40 harder?",
            "Young and normal ECGs show more sinus arrhythmia (respiratory RR variation), reducing periodicity.",
        ),
    ],
)

# =============================================================== 22 Lead-level
s = slide(
    "Lead-level analysis",
    "Errors scale with lead amplitude: aVL easiest, V2–V3 hardest; the final model is best on every lead",
    "Source: recomputed from OOF forecasts of the final configuration (hidden 128), folds 1–9, 3-seed ensembles per fold",
)
s.image(FIG / "leads.png", 0.45, 1.6, 12.4, 3.9)
s.table(0.45, 5.6, [2.6, 3.2, 3.2, 3.4], [
    ["Final model", "Most predictable", "Least predictable", "vs L1 CNN-LSTM"],
    ["MAE (mV)", "aVL 0.053 · aVF 0.058 · aVR 0.062", "V3 0.128 · V2 0.127 · V4 0.117", "lower on 12/12 leads (V6: tie)"],
], size=12, row_h=0.45)  # fmt: skip
notes(
    s,
    "Show where in the 12-lead ECG the errors are.",
    [
        "Precordial leads V2-V4 have roughly twice the amplitude of limb leads (Phase 2), and absolute errors follow amplitude. Relative to amplitude, leads are much more similar.",
        "The final model improves on the L1 CNN-LSTM on every lead; the largest absolute gains are in V1-V4 (≈ 2-3 µV), V6 is a tie.",
    ],
    [
        "sn4 is worse than the flat line on every lead under MAE — the pointwise penalty for mistimed beats — but better on shape, which the gatekeepers capture."
    ],
    [
        "Per-lead analysis was recomputed for the frozen hidden-128 configuration so this slide matches the final model exactly.",
        "Gate values are slightly higher for limb and right-precordial leads (0.27-0.29) than V4-V6 (0.22-0.24) (Phase 15, hidden-256 variant).",
        "Error also grows with horizon: from 0.062 mV in the first 100 ms to 0.101 mV in the last 100 ms (Phase 15).",
    ],
    [
        (
            "Should the model predict 8 leads and derive 4?",
            "It guarantees lead consistency but cost +2.9 µV MAE for the CNN-LSTM (Phase 9); for the gated model it could fix the lead-consistency issue on the next slide — future work.",
        ),
        (
            "Is there a normalised view?",
            "Not computed for this slide; amplitude-normalised errors were not part of the pre-registered analysis.",
        ),
    ],
)

# =============================================================== 23 Plausibility
s = slide(
    "Physiological plausibility",
    "Right number of beats in roughly the right places — but damped, and limb-lead relations break",
    "Sources: Phase 13 (OOF, hidden-256 variant); examples recomputed for the final model (fold 9 OOF)",
)
s.image(FIG / "examples.png", 0.45, 1.55, 7.6, 3.2)
s.table(8.3, 1.6, [2.4, 1.0, 0.85, 0.75], [
    ["Measure", "Final", "L1", "sn4"],
    ["Beats detected / true", "1.00", "0.25", "1.00"],
    ["No beat in forecast", "3.9%", "42%", "1.4%"],
    ["Timing ≤ 50 ms", "35%", "0.2%", "44%"],
    ["Peak-to-peak / truth", "0.28", "0.20", "0.59"],
    ["ST level error (mV)", "0.054", "0.083", "0.058"],
    ["Limb violation > 0.01 mV", "19%", "3%", "0%"],
    ["Out of history range", "0.15%", "0.13%", "0.01%"],
], size=10.5, row_h=0.4)  # fmt: skip
key(s, 0.45, 4.95, 12.4, 1.85, "Success (regular rhythm): QRS complexes placed correctly at reduced amplitude, slow waves tracked.  "
    "Failure (AF): irregular beats cannot be copied; the gate closes and the forecast reverts to a slow-wave average.  "
    "The per-lead gate mixes sn4 in different proportions per lead, so Einthoven/Goldberger relations break in 19% of records (median 7 µV).", 12.5)  # fmt: skip
notes(
    s,
    "Judge forecasts as ECGs, not just as numbers.",
    [
        "The final model produces the right number of beats (median ratio 1.0; L1: 0.25) and a third of records within 50 ms, but at about 28% of true amplitude — a damped mixture of sn4 and the L1 mean forecast.",
        "ST level is the one wave measure where the model beats sn4 (0.054 vs 0.058 mV), consistent with its slow-wave tracking.",
        "Lead consistency got worse (19% vs 3%) because the gate acts per lead independently. Violations are small but systematic.",
    ],
    [
        "These are signal-level checks; no cardiologist review or clinical validation was performed.",
        "Forecasts stay inside the history's range more often than the truth does — there are no implausible excursions.",
    ],
    [
        "Timing bands (D6.6) for the final model: 21% ≤ 25 ms, 13% 25-50 ms, 33% 50-150 ms, 32% > 150 ms; L1: 95% > 150 ms.",
        "On the sealed fold 10, 36% of the final model's forecasts were within 50 ms.",
    ],
    [
        (
            "Is 28% amplitude acceptable?",
            "Not for clinical use; it is the L1-optimal hedge under timing uncertainty. Removing the gate restores amplitude but costs 16 µV MAE (slide 25).",
        ),
        (
            "How would you fix lead consistency?",
            "Share the gate across limb leads, or predict 8 independent leads and derive the other 4 exactly.",
        ),
    ],
)

# =============================================================== 24 Explainability
s = slide(
    "Explainability",
    "The model learned when to trust the repeated beats — and relies on the last second of history",
    "Gradient saliency and decomposition: 6 fold-9 models (hidden 256, trained on folds 1–8), 400 fold-9 records; ablations: Phase 18",
)
s.image(FIG / "explain_decomposition.png", 0.45, 1.55, 12.4, 2.5)
s.image(FIG / "explain_saliency.png", 0.45, 4.15, 6.4, 2.2)
s.text(7.05, 4.1, 5.9, 2.9, [
    "**Saliency:** 70% within the last 1 s of history; at QRS (±50 ms) only half the level elsewhere → slow waves drive the network",
    "**Gate:** mean 0.26; 0.36 → 0.16 across the horizon; tracks periodicity (ρ = 0.47)",
    "**Ablations:** no gate → +16 µV MAE (full sn4 beats); sn4 as input only → timing 264 ms (= no sn4) — slide 25",
    "SHAP not computed (8,400-dim input); attention N/A (final model has none)",
], size=12.5)  # fmt: skip
notes(
    s,
    "Explain what the model learned, using methods appropriate to this architecture, and be explicit about what was not done.",
    [
        "Decomposition (top): the forecast is literally gate × sn4 + correction. In the example, the gated term carries the QRS complexes (shrunk), while the correction carries the slow baseline drift.",
        "Saliency (bottom left): gradient of the first 500 ms of the lead-II forecast with respect to the history. Seventy percent of the residual-channel saliency falls in the final second, and saliency at QRS complexes is about half that elsewhere: the network's own pathway relies on slow waves and the recent level, while beat structure enters through the sn4 skip path.",
        "Ablations are the causal evidence: give sn4 only as an input channel (no skip path) and the network ignores it (timing 264 ms, like no sn4). Remove the gate and beats come through at full amplitude, matching sn4 on shape but costing 16 µV MAE.",
    ],
    [
        "Requested methods not used, and why: SHAP — KernelSHAP over 8,400 input values per record is computationally infeasible on CPU, and gradient saliency is the standard alternative for differentiable sequence models. Attention maps — the final model has no attention; the Transformer was not carried into the final phase."
    ],
    [
        "The explainability analysis is descriptive and post hoc (added for this presentation), using out-of-sample fold-9 models of the hidden-256 variant, which shares the final model's mechanism.",
        "Baseline channels receive 48% of the total saliency, consistent with the importance of the slow component.",
    ],
    [
        (
            "Does the gate know when sn4 is wrong?",
            "Only partly: its correlation with sn4's own error is weak (ρ = −0.11), versus 0.47 with history periodicity. That is why the model is slightly worse than L1 for irregular rhythms.",
        ),
        (
            "Is saliency reliable?",
            "Gradient saliency is noisy per record; the conclusions here are averages over 400 records and 6 models, and are corroborated by the ablations.",
        ),
    ],
)

# =============================================================== 25 Ablations
s = slide(
    "RQ7 · Ablations",
    "The sn4 skip path carries the timing skill; the gate trades beat shape for MAE",
    "Source: Phase 18 report — fold 9, 6 seeds per arm, compare_runs (ablation − final, 95% CI). Analysed the hidden-256 variant; ensemble MAE",
)
s.table(0.45, 1.65, [3.4, 1.15, 2.1, 1.1, 1.15, 1.1, 1.0, 1.4], [
    ["Arm", "MAE", "ΔMAE vs final (µV)", "DTW", "Template", "Timing", "≤ 50 ms", "Beats final?"],
    ["final (gate + sn4 skip + correction)", "0.0851", "–", "0.0694", "0.0934", "126", "30%", "–"],
    ["no gate (sn4 + correction)", "0.1017", "+16.1 [+14.6, +17.8]", "0.0610", "0.0643", "89", "43%", "✗ (MAE)"],
    ["sn4 as input only (no skip)", "0.0893", "+4.8 [+3.8, +5.9]", "0.0788", "0.1290", "264", "0.3%", "✗"],
    ["no sn4 (L1 CNN-LSTM)", "0.0874", "+2.9 [+2.0, +3.9]", "0.0781", "0.1288", "267", "0.4%", "✗"],
    ["raw input (no 0.67 Hz decomposition)", "0.0841", "−1.1 [−1.8, −0.5]", "0.0684", "0.0915", "120", "31%", "✓"],
    ["sn4 alone (no network)", "0.1220*", "+33.8 [+32.1, +35.5]", "0.0799", "0.0642", "89", "43%", "–"],
], size=11, row_h=0.43, highlight=(5,))  # fmt: skip
s.text(0.45, 4.8, 12.4, 1.4, [
    "**Skip path essential:** given sn4 only as an input channel, the network ignores it (timing 264 ms ≈ no sn4)",
    "**Gate = amplitude hedge:** without it, beats keep full amplitude (template and timing = sn4, best DTW) but MAE is 16 µV worse",
    "**Input decomposition:** raw input was better on fold 9 (not CV-confirmed); the frozen model kept the decomposition (D18.1)",
], size=12.5)  # fmt: skip
s.text(0.45, 6.3, 12.4, 0.35, ["MAE in mV (6-run ensemble; *single forecast), template in mV, timing in ms; lower is better."],
       size=10.5, color=GREY, bullet=False)  # fmt: skip
notes(
    s,
    "Give the causal evidence for which parts of the final model matter.",
    [
        "Each arm removes one component and is compared with the full model on the same fold-9 patients with 6 runs per arm.",
        "Removing the skip path (sn4 as an extra input only) or removing sn4 altogether gives the same result: timing around 265 ms. The timing skill comes entirely from passing sn4 through to the output.",
        "Removing the gate (sn4 + correction) gives the best shape and timing of any learned arm, at a 16 µV MAE cost: the gate's job is to damp beats when their timing is uncertain, which is what L1 rewards.",
    ],
    [
        "The 'no gate' arm is a legitimate alternative final model if morphology matters more than MAE — it is still 18 µV better than sn4 alone on MAE.",
        "Raw input beating the decomposition contradicts D5.11, which was decided on the reference CNN; it is a single-fold result and was not acted on before the test.",
    ],
    [
        "Phase 18 analysed the hidden-256 configuration (2.68 M parameters); the mechanism is identical in the hidden-128 final model.",
        "Capacity is not the explanation: 'sn4 as input only' has more parameters than the L1 CNN-LSTM but no timing gain.",
    ],
    [
        (
            "Why not switch to raw input after this result?",
            "It was a single-fold comparison found after the configuration was pre-specified; switching would have needed CV confirmation. It is listed as follow-up work.",
        ),
        (
            "Which ablation would you show a sceptic first?",
            "'sn4 as input only': it rules out the explanation that the model simply has access to more information.",
        ),
    ],
)

# =============================================================== 26 Compute
s = slide(
    "Compute efficiency",
    "The final model trains faster than three of four families; inference cost is dominated by computing sn4",
    "†Training min/run: measured during experiments (4 CPU threads, shared 40-core machine). RAM and inference: measured for this deck, random weights, 4 threads",
)
s.image(FIG / "compute.png", 0.45, 1.6, 12.4, 2.9)
s.table(0.45, 4.7, [2.8, 9.6], [
    ["Note", "Detail"],
    ["‡ Inference", "network only; the final model also needs sn4 (R-peak detection + period estimation): ≈ 7.5 ms CPU per record"],
    ["Final model", "1.17 M params; ≈ 43 min/run on CPU; 548 MB peak RAM"],
    ["Objective runs", "Phase 10 stages A–C: 550 min recorded training time; earlier phases excluded"],
    ["Not measured", "GPU throughput; energy"],
], size=11.5, row_h=0.42)  # fmt: skip
notes(
    s,
    "Give an honest cost picture for every model, including the final one.",
    [
        "Training time per run is the practical bottleneck on CPU: PatchTST (160 min) and the Transformer (110 min) are 2-3× the CNN-LSTM.",
        "The gated final model took about 43 min/run versus 53 min/run for the Phase 9 CNN-LSTM; this descriptive comparison is affected by shared-machine contention.",
        "Inference is sub-millisecond per record for all networks; for the final model, the hand-crafted sn4 feature (R-peak detection with NeuroKit2 + autocorrelation period search) dominates at ~7.5 ms CPU per record — still real-time capable.",
    ],
    [
        "Memory and inference were measured for this presentation in fresh processes (training step at batch 64, inference at batch 512). Training times come from the experiment logs and include contention on a shared machine, so treat them as relative."
    ],
    [
        "The CNN-LSTM has the slowest training step (1.0 s) because its LSTM runs over 175 steps with hidden 128 and two layers; the LSTM family is cheaper per step but needs more epochs per unit of accuracy.",
        "No GPU was available; all conclusions about cost are CPU-specific.",
    ],
    [
        (
            "Could the sn4 computation be learned?",
            "Possibly, but the ablation shows the network does not discover it from input alone; making it differentiable is future work.",
        ),
        (
            "Is the final model deployable?",
            "Computationally yes (≈ 8 ms/record on CPU); clinically no validation has been done.",
        ),
    ],
)

# =============================================================== 27 Test
s = slide(
    "Held-out test (fold 10, run once)",
    "The frozen final model reproduced its cross-validation result on 1,904 unseen patients",
    "Source: Phase 17 report; models frozen with SHA-256 hashes before unsealing; compare_runs on 6 runs per arm",
)
s.image(FIG / "test.png", 0.45, 1.6, 12.4, 3.2)
s.table(0.45, 4.95, [2.6, 2.3, 1.9, 2.4, 2.0, 1.2], [
    ["Final − reference", "ΔMAE (µV)", "ΔDTW (µV)", "ΔTemplate (µV)", "ΔTiming (ms)", "Beats?"],
    ["vs flat", "−32.4 [−34.4, −30.7]", "−50.5", "−12.6", "−179", "✓"],
    ["vs L1 CNN-LSTM", "−1.7 [−2.6, −0.9]", "−5.9", "−40.6", "−145", "✓"],
    ["vs sn4", "−35.6", "−11.7", "+27.0 [+23.5, +30.4]", "+31.6", "✗"],
], size=11.5, row_h=0.4, highlight=(2,))  # fmt: skip
notes(
    s,
    "Present the single unbiased estimate of the final model.",
    [
        "Fold 10 was sealed in code from Phase 0 until the final model was frozen, then evaluated once (lock file). Results match the pre-registered expectations from CV: MAE 0.0850 (expected ≈ 0.085), timing 110 ms (expected ≈ 109).",
        "The final model beats the flat line and the L1 CNN-LSTM on MAE and every gatekeeper; it remains worse than sn4 on morphology and timing.",
    ],
    [
        "The MAE gain over L1 is small in absolute terms (1.7 µV, 2%), consistent with CV (1.6 µV); the gatekeeper gains are large (template −41 µV, timing −145 ms).",
        "The final model reverted from hidden 256 to hidden 128 after CV, before fold 10 was opened (user decision D12.1).",
    ],
    [
        "Fold 10 has 2,198 records after the same cleaning rules; like fold 9 it is fully human-validated.",
        "A dry run of the test script on fold 9 verified the code path and the hash checks beforehand; its numbers were not used.",
    ],
    [
        (
            "Could anything have leaked from fold 10 into design?",
            "Not even fold-10 metadata were examined before Phase 17; the loader refused fold 10 unless explicitly enabled.",
        ),
        (
            "Is one fold enough?",
            "It is the official PTB-XL test protocol; external validation on another dataset remains future work.",
        ),
    ],
)

# =============================================================== 28 Final comparison
s = slide(
    "Final model comparison",
    "Only the sn4-gated CNN-LSTM is better than the L1 models on every criterion; sn4 still owns morphology",
    "Accuracy/shape/timing: fold 9 single-model mean of 6 seeds (comparable across all rows). Generalisation: 9-fold CV MAE / fold 10. Costs: slide 26",
)
s.table(0.45, 1.7, [2.75, 1.05, 1.15, 1.0, 1.95, 1.2, 1.25, 1.05], [
    ["Model", "MAE", "Template", "Timing", "CV MAE / fold 10", "Train min", "Infer ms/rec", "Params"],
    ["Flat line", "0.1161", "0.105", "298", "0.1097 / 0.1207", "–", "≈ 0", "0"],
    ["sn4 (repeat beats)", "0.1218", "0.064", "89", "0.1167 / 0.1239", "–", "≈ 7.5", "0"],
    ["CNN (reference)", "0.0961", "0.105", "259", "0.0940 / –", "8", "0.13", "1.85 M"],
    ["LSTM", "0.1030", "0.105", "254", "–", "33", "0.37", "0.25 M"],
    ["CNN-LSTM (L1)", "0.0911", "0.129", "267", "0.0866 / 0.0861", "53", "0.49", "0.72 M"],
    ["PatchTST* (reduced)", "0.0963", "0.116", "276", "–", "160", "0.68", "0.74 M"],
    ["Transformer", "0.0944", "0.109", "283", "0.0916 / –", "110", "0.61", "1.18 M"],
    ["CNN-LSTM + STFT loss", "0.0935", "0.091", "112", "0.0873 / –", "59", "0.49", "0.72 M"],
    ["sn4-gated CNN-LSTM (final)", "0.0890", "0.093", "125", "0.0850 / 0.0850", "43", "0.52 + 7.5", "1.17 M"],
], size=12, row_h=0.43, highlight=(9,))  # fmt: skip
s.text(0.45, 6.15, 12.4, 0.8, ["MAE and template in mV, timing in ms, lower is better. Fold-10 values are 6-run ensembles; "
                               "the final model's fold-9 single-model numbers are higher than its ensemble/CV numbers by design."],
       size=11, color=GREY, bullet=False)  # fmt: skip
notes(
    s,
    "One table that justifies the final choice across accuracy, plausibility, generalisation and cost.",
    [
        "Reading guide: the best L1 architecture (CNN-LSTM) has the best MAE among L1 models but the worst morphology. The STFT loss fixes morphology but loses MAE. The sn4-gated model is the only one better than the L1 CNN-LSTM on MAE, template and timing simultaneously, and it generalises (CV and fold 10 agree at 0.0850).",
        "sn4 itself remains best on template MAE and timing, at a much worse MAE; no learned model dominates it.",
    ],
    [
        "Complexity: the final model has 1.6× the parameters of the plain CNN-LSTM but trains faster; its main extra cost is the sn4 feature at inference.",
        "The '–' entries are configurations that were not cross-validated or not tested on fold 10, by design (only finalists and the final model).",
    ],
    [
        "Fold-9 single-model values are used for the comparable columns because every model has them; ensembles improve all learned models by 1-4 µV MAE.",
        "The L1 CNN-LSTM's fold-10 value comes from its 6-seed reference ensemble trained alongside the final model.",
    ],
    [
        (
            "Why not choose the STFT model for better shape?",
            "It costs MAE against L1 (0/9 folds better), failing the MAE-primary rule; the gated model is better than it on every metric anyway.",
        ),
        (
            "Is the final model 'best'?",
            "It is the best under the pre-registered criterion among the models tested; it is not better than sn4 on beat morphology or timing.",
        ),
    ],
)

# =============================================================== 29 Decisions
s = slide(
    "Key research decisions",
    "Twelve decisions determined the outcome; each is logged with alternatives and evidence",
    "Source: DECISION_LOG.md (D0–D17)",
)
s.table(0.45, 1.65, [0.75, 3.3, 4.6, 3.75], [
    ["ID", "Decision", "Evidence", "Why it mattered"],
    ["D0.8", "Keep 7 → 2.9 s; reformulate evaluation", "Flat beats repeated beats under MAE; oracle −34%", "Prevented a blur-rewarding benchmark"],
    ["D3.1/3.4", "Exclude 30 defects; trim last 10 samples", "Artefact up to 0.15 mV vs 500 Hz originals", "Clean target, all folds alike"],
    ["D4.x", "Patient-level folds; fold 10 sealed in code", "Record splits: 20% patient overlap, 3.2% bias", "Leakage-free estimates"],
    ["D6.2", "MAE + 3 gatekeepers rule", "All pointwise metrics fail the 50 ms axiom", "Changed the Phase 9 verdict"],
    ["D8.3", "6 runs/arm, run + patient bootstrap", "False wins 41% → 2.8%", "Trustworthy sub-µV verdicts"],
    ["D5.11", "Adopt 0.67 Hz decomposition input", "−0.34 µV, replicated", "Small; kept for consistency"],
    ["D9.3", "No best family under L1", "MAE gains with worse morphology", "Pointed to the objective"],
    ["D10.0", "Objectives on CNN-LSTM only", "Transformer 2× cost, CPU only", "Disclosed scope limit"],
    ["D10.2", "Adopt sn4-gated CNN-LSTM", "Beats L1 and flat, 9/9 folds, all metrics", "The key modelling result"],
    ["D12.1", "Revert hidden 256 → 128", "CV: template +1.3 µV, timing +5 ms", "Multi-fold over single-fold"],
    ["D17", "Open fold 10 once", "Hash-verified frozen models", "Unbiased final estimate"],
], size=10.5, row_h=0.41)  # fmt: skip
notes(
    s,
    "Summarise the decision trail that a committee would want to audit.",
    [
        "Every decision in DECISION_LOG.md lists alternatives, evidence, the selected option, the reason and its status (accepted, provisional, pending, and whether it was a user decision).",
        "User decisions at phase boundaries included: trimming the artefact, adopting the decomposition, fold-9 screening with multi-fold confirmation, the Phase 9 epoch amendment, the Phase 10 scope, reverting the hidden size, and opening fold 10.",
    ],
    [
        "Two decisions deliberately overrode a single-fold signal in favour of multi-fold evidence (D12.1) or kept a pre-specified choice despite a single-fold contrary result (D18.1). Both are consistent with the 'confirm before adopting' principle."
    ],
    [
        "Deviations from pre-registration are labelled as amendments with dates: Phase 9 epoch cap (40 → 100), Phase 10 family scope, Phase 16 final configuration."
    ],
    [
        (
            "Which decision had the biggest impact?",
            "D6.2 (the gatekeeper rule): without it, the study would have concluded that the L1 CNN-LSTM solves the task.",
        ),
        (
            "Which decision is most debatable?",
            "D10.0 — dropping the Transformer from the objective experiments to save compute.",
        ),
    ],
)

# =============================================================== 30 Conclusions
s = slide(
    "Conclusions",
    "The objective, not data, preprocessing or architecture, was the bottleneck — gating repeated beats unlocks it",
)
s.table(0.45, 1.65, [3.0, 9.4], [
    ["Question", "Answer"],
    ["What worked?", "A learned per-step, per-lead gate on the causal sn4 forecast + correction: −1.7 µV MAE, −41 µV template, −145 ms timing vs L1 on fold 10"],
    ["What did not?", "More data (≤ 0.6% beyond 60%), filters/scalers/augmentation (< 1 µV), derivative loss, 8-lead output, larger LSTM (fold-9 gain not confirmed)"],
    ["Best model", "sn4-gated CNN-LSTM, hidden 128, 6-seed ensemble — fold 10 MAE 0.0850 mV, timing 110 ms, 36% of beats ≤ 50 ms"],
    ["Best preprocessing", "Raw-mV 12-lead target; causal 0.67 Hz baseline + residual input (tiny gain; raw input competitive)"],
    ["Largest contributors", "sn4 skip path (vs none: +2.9 µV MAE, +35 µV template, +140 ms timing) > architecture (CNN-LSTM vs CNN: −7.4 µV) ≫ preprocessing, data size"],
    ["Still unsolved", "Matching sn4 on morphology and timing; irregular rhythms (AF); amplitude damping (≈ 28%)"],
], size=12.5, row_h=0.68)  # fmt: skip
notes(
    s,
    "Answer the five concluding questions directly.",
    [
        "What worked: conditioning on a causal, physically meaningful forecast (repeat the recent beats) and letting the network decide how much to trust it. It is the first change in the study that improves accuracy and plausibility together, and it held on unseen patients.",
        "What did not: every lever that only changes the input or the amount of data moved MAE by less than 1 µV and did nothing for timing.",
        "Contribution ranking is based on ablations and paired comparisons: removing sn4 costs the most on every metric; architecture choice mattered for MAE; preprocessing and data size were marginal.",
    ],
    [
        "Be precise about 'best': best under the pre-registered criterion among the models tested — not superior to sn4 on beat shape and timing, and not clinically validated."
    ],
    [
        "Methodological contributions that transfer beyond this task: axiom-tested metric selection with gatekeepers; run-and-patient bootstrap with empirical calibration; sealed test fold with hash-verified freeze."
    ],
    [
        (
            "Is the improvement over L1 clinically meaningful?",
            "Unknown; the 2% MAE gain is small, the gains in morphology and timing are large in signal terms, but no clinical endpoint was studied.",
        ),
        (
            "What would you do differently with a GPU?",
            "Run full-size PatchTST/Transformer with the gate, and timing-tolerant objectives such as soft-DTW.",
        ),
    ],
)

# =============================================================== 31 Limitations
s = slide(
    "Limitations and threats to validity",
    "What the evidence does not show — and how each threat was contained",
    "Sources: FINAL_REPORT.md §6 and §8; Phases 9, 10, 12, 16, 17 reports; README (reproducibility)",
)
s.table(0.45, 1.65, [2.9, 5.4, 4.1], [
    ["Threat", "Detail", "Containment / consequence"],
    ["One held-out fold, one dataset", "Fold 10 opened once; no external cohort or other devices", "Fold 10 now used; new models need a new test set"],
    ["Optimistic development numbers", "Selection on fold 9; only finalists cross-validated", "9-fold CV for finalists; fold 10 matched CV (0.0850)"],
    ["Analysis model ≠ final model", "Phases 12–15 and 18 analysed hidden 256; final is hidden 128 (D12.1)", "≈ 1 µV MAE apart; same mechanism; hidden 128 CV'd in Phase 10"],
    ["CPU-only compute", "PatchTST reduced; one configuration per family (lr only); objectives on CNN-LSTM only (D10.0)", "Family ranking conditional on these sizes"],
    ["Detector-dependent metrics", "Timing/template use lead-II R-peaks (NeuroKit2); 100 Hz cannot resolve QRS duration or QT", "Lead II missing breaks timing (slide 20)"],
    ["Approximate statistics", "CV p-values with overlapping training sets; no multiplicity correction; rhythm strata from PTB-XL statements", "Pre-registered comparisons; run + patient bootstrap CIs"],
    ["Point forecasts, no clinical endpoint", "Damped beats (≈ 28% amplitude) are the L1 hedge", "Not a clinical tool; probabilistic forecasts as future work"],
], size=10.5, row_h=0.52)  # fmt: skip
key(s, 0.45, 5.95, 12.4, 0.85, "Reproducibility safeguards: locked uv environment, validated config + hash, provenance manifest per run, "
    "bitwise reruns in every phase, SHA-256-frozen final models, 98 unit tests.", 12.5)  # fmt: skip
notes(
    s,
    "State the limits of the conclusions before proposing future work, so that each future direction answers a named limitation.",
    [
        "The most important limitation is external validity: everything comes from PTB-XL, and only one held-out fold exists. Fold 10 cannot be reused for new model selection.",
        "Several post-test analyses (robustness, error analysis, ablations) were run on the hidden-256 variant. This is disclosed on every affected slide; the two configurations differ by about 1 µV MAE and share the same mechanism.",
    ],
    [
        "CPU-only compute shaped the study: it limited family sizes, hyperparameter search and the objective study to one family. Conclusions about architectures are conditional on those sizes.",
        "The beat-based gatekeepers depend on R-peak detection in lead II. They were validated in Phase 6 against controlled probe forecasts, but they inherit detector errors and the 100 Hz resolution limit.",
    ],
    [
        "Reproducibility: each run writes its resolved config, provenance (git commit, lockfile, packages, hardware, dataset fingerprint), metrics and log; reruns were bitwise identical on the same hardware and thread count, and one such check exposed the Phase 7 seeding defect.",
        "Across machines, results are only statistically reproducible, which is why all verdicts rest on multiple seeds.",
    ],
    [
        (
            "How confident are you in the fold-10 number?",
            "It is an unbiased single estimate for these patients (1,904) and matches the CV mean exactly; it does not tell us about other hospitals or devices.",
        ),
        (
            "Which limitation would change the conclusion if removed?",
            "Probably compute: full-size attention models with the gate, or soft-DTW objectives, could change the family ranking — but not the finding that the objective, not the input, was the bottleneck.",
        ),
    ],
)

# =============================================================== 32 Future work
s = slide("Future work", "Five directions that follow directly from the evidence")
s.table(0.45, 1.65, [4.0, 5.0, 3.4], [
    ["Direction", "Evidence that motivates it", "Expected benefit"],
    ["Periodicity-aware gating / fallback for irregular rhythms", "Gain only in periodic histories; AF no better than L1; gate weakly tracks sn4 error (ρ −0.11)", "Remove losses in least-periodic 60%"],
    ["Lead-consistent output (shared limb gate or 8 leads + derived)", "Limb-lead violations 19% vs 3% for L1", "Consistency by construction"],
    ["Probabilistic / multi-hypothesis forecasts", "Damping (≈ 28% amplitude) is the L1 hedge under timing uncertainty", "Sharp beats with calibrated uncertainty"],
    ["Timing-tolerant objectives on GPU (soft-DTW; spectral + gate)", "Spectral λ 0.3 gave best morphology in tuning; soft-DTW deferred on CPU", "Close the gap to sn4 on shape"],
    ["Full-size attention models + external validation", "PatchTST reduced for CPU; one held-out fold, one dataset", "Fairer family comparison; generalisation"],
], size=12, row_h=0.78)  # fmt: skip
notes(
    s,
    "Close with research directions grounded in specific findings, not a generic wish list.",
    [
        "Each direction is tied to a measured weakness: the periodicity dependence (Phase 15), the lead-consistency break (Phase 13), amplitude damping (Phase 13/18), the objective trade-off (Phases 10-11), and compute-limited comparisons (Phase 9).",
        "The highest-value next step is probably the periodicity-aware gate: it targets the subgroup (irregular rhythms) where the current model is worse than the simpler L1 model.",
    ],
    [
        "Also worth re-checking with CV: raw input vs the adopted decomposition for the gated model (raw was 1.1 µV better on fold 9 only)."
    ],
    [
        "Excluded on purpose: more data and more preprocessing variants, since the evidence says they are marginal.",
        "Any new model must now be compared on a different held-out set, since fold 10 has been used once.",
    ],
    [
        (
            "Which direction is lowest cost?",
            "Lead-consistent output: a shared limb gate is a small architectural change and testable on CPU.",
        ),
        (
            "How would you evaluate probabilistic forecasts?",
            "With proper scoring rules (e.g. CRPS or energy score) alongside the existing gatekeepers applied to samples.",
        ),
    ],
)

deck.save(OUT)
print(f"saved {OUT} ({len(deck.slides)} slides)")
