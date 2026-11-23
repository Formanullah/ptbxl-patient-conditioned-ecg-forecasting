# Phase 2 pre-registration: dataset understanding

Written 2026-11-22 before the Phase 2 analysis ran. Already known from earlier phases
(disclosed): fold sizes, quality-flag rates, ages above 89 stored as 300, heart-rate
and rhythm summaries from Phase 0.

## Scope

- **Folds 1–9 only** (19,601 records). Fold 10 stays sealed: not even its metadata
  distributions are examined, so test-set label or demographic shifts cannot
  influence design choices.
- Raw 100 Hz signals in mV, without filtering. Preprocessing comes in Phase 5.
- Ages stored as 300 are treated as a separate "> 89" category, never as a number.

## Descriptive questions, and the decision each one informs

| # | Question | Informs |
|---|---|---|
| Q1 | How are records distributed over patients (records per patient; time between a patient's recordings)? | Patient-level aggregation and leakage control (Phases 4, 6, 12) |
| Q2 | What are the age, sex, height and weight distributions, and how much is missing? | Subgroup analyses (Phase 14); whether demographics are usable inputs |
| Q3 | What are the label distributions: superclasses, subclasses, rhythm and form statements, label likelihoods, labels per record, co-occurrence, human validation? | Stratified error analysis (Phase 14); rare-group sample sizes |
| Q4 | How are recordings distributed by year, site, device and fold? | Possible acquisition confounders |
| Q5 | What are the per-lead amplitude, offset and correlation statistics? | Scaling choice (Phase 5); lead-level analysis (Phase 15) |

## Quantitative checks with fixed thresholds

**R1 — Fold 9 is representative of folds 1–8 (validation-set fairness).**
- Pass if the absolute standardized mean difference (SMD) between fold 9 and folds
  1–8 is < 0.1 for every one of: age, sex, age > 89, the five diagnostic
  superclasses, SR, AFIB, the four quality flags, heart rate, and mean per-lead
  signal SD.
- 0.1 is the conventional threshold for negligible imbalance (Austin, 2009).

**R2 — The limb leads are linear combinations of two leads.**
- Pass if, for at least 99% of records, each of
  - III = II − I
  - aVR = −(I + II) / 2
  - aVL = I − II / 2
  - aVF = II − I / 2

  holds with RMS residual ≤ 0.01 mV (10 ADC steps).
- If R2 passes, only 8 of the 12 leads carry independent information. That bears
  directly on the model output design and on Phase 13's lead-consistency checks.

**R3 — Repeat recordings matter.**
- Pass if at least 10% of records come from patients with more than one record.
- If R3 passes, patient-clustered statistics (already used in Phase 0) are
  mandatory, not optional.
