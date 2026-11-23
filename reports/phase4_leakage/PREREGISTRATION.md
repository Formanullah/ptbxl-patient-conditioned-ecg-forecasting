# Phase 4 pre-registration: data leakage prevention

Written 2026-11-22 before the Phase 4 analysis ran. Data: cleaned folds 1–9 (19,571
records, 990 samples each, D3.1/D3.4).

For fold 10, only `ecg_id`, `patient_id` and `strat_fold` are read, to verify patient
disjointness. Its signals, labels and demographics stay sealed.

## Leakage channels examined

| Channel | Mechanism in this task |
|---|---|
| Patient-level | Other recordings of the same patient in training let a model memorize the patient |
| Fold-level | Patient IDs shared across train/validation/test folds |
| Repeated measurements | 23% of records come from repeat patients (Phase 2) |
| Metadata | PTB-XL statements, flags and reports describe the *whole 10 s*, including the forecast window |
| Preprocessing, across records | Statistics (scalers, filter design, thresholds) fitted on validation or test data |
| Preprocessing, within a record | Non-causal operations (zero-phase filtering, whole-record normalization, detrending, R-peak detection on the full record) let the history contain information about its own future |

## Split strategies compared (on cleaned folds 1–9)

| Strategy | Repeats |
|---|---|
| R: random record-level 80/20 | seeds 42, 123, 999 |
| SR: label-stratified record-level 80/20 (diagnostic superclass pattern) | seeds 42, 123, 999 |
| P: patient-level random 80/20 | seeds 42, 123, 999 |
| OF: official patient-level, label-stratified folds (train 1–8, validate 9) | 1 |
| KF vs GKF: 5-fold record-level K-fold vs 5-fold patient-level GroupKFold | 5 folds each |

**Memorization probe.** A k-nearest-neighbour forecaster (k = 1 and 5). It matches the
2 s before the last R peak of the query's history (R peaks detected on the history
only) against R-peak-anchored segments of training records, then copies the
neighbour's continuation, level-adjusted. It can only do well by finding similar
recordings, which makes it a direct test for memorization.

Measured per split:
- patient-macro MAE (mV, 290 × 12 target);
- share of validation queries whose nearest neighbour is the same patient;
- share of validation records whose patient also appears in training;
- maximum |SMD| of superclasses and device families between train and validation.

## Hypotheses

**H4.1 — Structural (must hold):**
- official train (1–8), validation (9) and test (10) folds share no `patient_id`;
- `ecg_id` is unique;
- `patient_id` is a credible identity: sex is constant within ≥ 99% of multi-record
  patients, and |Δage − Δrecording-years| ≤ 1.5 for ≥ 95% of same-patient record
  pairs (ages > 89 excluded).

**H4.2 — Patient overlap biases validation optimistically.**
- With k = 1, the record-level random split (R) gives a lower patient-macro kNN MAE
  than the patient-level random split (P).
- The mean paired difference over the three seeds must exceed twice its SD.
- Magnitude and same-patient neighbour rate are reported either way.

**H4.3 — Full-record labels leak the future.**
- Take records whose history (0–7 s) has regular RR intervals (all within 20% of
  their median).
- Among them, the extra-beat annotations (statements PVC or PAC, or a non-empty
  `extra_beats` field) are at least 2× as frequent when the target window
  (7.0–9.9 s) contains an RR interval deviating > 20% from the history median, as
  when it does not.

**H4.4 — Non-causal preprocessing leaks the future into the history.**
- Predict each lead's mean over the target window as the value at sample 699 of a
  0.67 Hz low-pass of the signal.
- If the low-pass is zero-phase over the whole record, the MAE of this prediction is
  ≥ 10% lower than with a causal low-pass of the same order and cutoff.

## Decision rule

- The split must be **patient-level** (no patient in two partitions).
- Among patient-level options, the **official folds** are kept unless GroupKFold shows
  a lower leakage risk. Holdout vs cross-validation is a Phase 8 question.
- Every channel that shows leakage, or cannot be ruled out, gets a code-level guard
  with tests.
