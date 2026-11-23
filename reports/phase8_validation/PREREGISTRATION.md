# Phase 8 pre-registration: validation strategy

Written 2026-11-22 before any Phase 8 computation. Fold 10 stays sealed: neither its
signals nor its metadata are read.

## Questions and their origin

| Question | Origin |
|---|---|
| Q1. Is fold 9 a suitable model-selection fold despite its device shift? | D2.3 (provisional) |
| Q2. Holdout (fold 9) or cross-validation for model selection? | D4.1 |
| Q3. How should comparisons account for training randomness? | D7.4 |

**Strategies considered.** Record-level K-fold is excluded: it leaks patients (Phase 4,
D4.1). GroupKFold by patient is represented by the **official folds**, which are a
fixed, published patient-grouped partition. Random patient groups would mix the
human-validated records (folds 9/10 by design) differently from the benchmark.
**Nested CV is assessed analytically, not run.** Its purpose is an unbiased performance
estimate when there is no independent test set. Here the sealed fold 10 provides that
estimate. Its cost is 9 outer × 8 inner trainings per configuration and seed.

## E1 — Fold 9 representativeness (Q1)

Groups: F9 = fold 9 (100% human-validated); V = human-validated records of folds 1–8
(11,662); N = the other records of folds 1–8 (5,730). Variables: indicator of each
device family (first token of `device`), baseline-drift flag, and baseline difficulty
from Phase 6: flat MAE, `sn4` MAE and `sn4` beat timing per record. Standardized mean
difference (SMD) per variable.

Known before this phase (Phase 2): fold 9 differs from folds 1–8 mainly in device mix
(CS100 SMD −0.89), and the drift flag occurs mostly on human-validated records. Phase 2
did not compare device mix or difficulty between V and N.

- **H8.1 — The fold-9 shift is a consequence of human validation.** max |SMD|(F9 vs V)
  < 0.1 over all variables, while max |SMD|(F9 vs N) ≥ 0.2.
- **Implication.** If accepted: fold 10, also human-validated by design, is expected
  to share fold 9's distribution, so fold 9 is the right selection fold. If rejected:
  fold 9's shift may be specific to that fold, which favours CV over folds 1–9 for
  selection.

## E2 — Holdout vs cross-validation (Q2)

**9-fold CV over the official folds 1–9.** Each fold k is held out once. Training uses
the other eight folds minus a 20% internal patient holdout (seed 0, as in Phases 5/7)
for early stopping and λ selection.

Models (6):
- flat and `sn4` (non-learned; Phase 6 scores);
- ridge (raw);
- CNN, raw input;
- CNN, decomposition input (adopted, D5.11);
- CNN, decomposition, trained on 20% of the pool's patients (subset order = run seed,
  as in Phase 7).

CNNs use seeds 42/123/999 and 3-seed ensembles. The fold-9 runs are the Phase 5/7 runs
(bitwise reproducible, so they are reused from cache).

Measures per model and fold: Phase 6 suite, patient-macro. Between-fold SD;
per-fold paired ΔMAE for all 15 pairs; Kendall τ between each single fold's ranking and
the CV-mean ranking (MAE).

- **H8.2 — Fold 9 holdout suffices for model selection.** (a) Every pair whose CV-mean
  ΔMAE is significant (paired t-test over the 9 folds, two-sided p < 0.05) has the
  same sign on fold 9. (b) Kendall τ(fold 9 ranking, CV-mean ranking) ≥ 0.8 on MAE,
  i.e. at most one discordant pair.
- **H8.3 — Absolute levels are fold-dependent.** For every learned model, fold 9's MAE
  exceeds the mean of folds 1–8 by more than 2 SD of folds 1–8.

## E3 — Training randomness and the comparison protocol (Q3)

Runs on the standard split (pool → fold 9), 18 seeds per configuration (42, 123, 999,
7, 2024, 31337, 101–112):
- A: CNN, decomposition, 100% of the pool;
- R: CNN, raw input, 100%;
- S: CNN, decomposition, 40% of the pool's patients (fixed subset: patient order
  seed 0), so that only training randomness varies.

Existing runs are reused where identical.

**Protocols** (MAE criterion of D6.2; gatekeepers only make a claim stricter and are
too costly to recompute for thousands of resamples — disclosed):

| ID | Arms | Inference |
|---|---|---|
| P0-3 | 3-seed ensembles | patient-cluster bootstrap (current D6.2) |
| P0-6 | 6-seed ensembles | patient-cluster bootstrap |
| P1 | two disjoint 3-seed ensembles per arm | both must pass P0-3 with the same sign (replication, as used for D5.11) |
| P2-3 | 3 single models | hierarchical bootstrap: seeds within each arm, then patients; statistic = difference of seed-mean patient-macro MAE |
| P2-6 | 6 single models | as P2-3 |

A claim is made when the 95% CI excludes 0 (either direction).

**Calibration.** 500 random draws of disjoint seed sets per scenario:
- **False-positive rate (FPR):** A vs A.
- **Power:** A vs S (moderate effect: Phase 7 found 40% vs 100% ≈ +1 µV in both
  replicates). Power for A vs R (small effect, ≈ 0.35 µV; D5.11) is reported only, to
  avoid selecting on the effect that motivated the question.
- **Wrong-sign claims** are reported.

- **H8.4 — The current protocol is anti-conservative.** FPR(P0-3) > 5%.
- **Selection rule (D8.3).** Among protocols with FPR ≤ 5%, choose the one with the
  highest power for A vs S. If two are within 5 percentage points, the one with fewer
  runs per arm wins.

## Decision rules

- **D8.1 (fold 9's role):** from H8.1 as stated under E1.
- **D8.2 (holdout vs CV):**
  - If H8.2 is accepted: select models on the fold-9 holdout in Phases 9–11 (1/9 of
    the cost), and run 9-fold CV only for the final shortlist (Phase 12 robustness).
  - If rejected: select shortlisted candidates with 9-fold CV.
- **D8.3 (comparison protocol):** the selection rule above. It replaces the MAE
  inference in D6.2; the gatekeeper logic is unchanged.

## Limitations fixed in advance

- 18 seeds per configuration estimate run-to-run variance with limited precision, and
  the 500 draws reuse them, so FPR estimates are correlated. They are a calibration,
  not an exact error rate.
- The A vs S effect is a data-size effect. Effects of architecture or objective
  changes may vary differently between runs.
