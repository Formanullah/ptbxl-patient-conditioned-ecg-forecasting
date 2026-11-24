# Phase 17 pre-registration: one-time test on fold 10

Written 2026-11-24, before fold 10 was ever loaded. **Not yet run: requires explicit
user approval** (user decision 2026-11-24: stop before Phase 17).

## Procedure (`research/phase17_test.py --unseal-fold-10-once`)

1. Verify the Phase 16 manifest hashes; refuse to run if a lock file exists; write the
   lock file *before* loading fold 10.
2. Load fold 10 with the same cleaning rules (defect exclusion D3.1, trim D3.4).
3. Compute `sn4`, flat and the true R peaks from each record; forecast with the 6 final
   and 6 L1 models.
4. Score with the Phase 6 suite (patient-macro).

## Primary analysis

`compare_runs` (D8.3) of the final model's 6 runs against flat, `sn4` and the 6 L1
runs; D6.2 decision rule. 6-run ensemble results reported alongside.

## Expectations (from the development folds)

Final beats flat and L1; it does not beat `sn4` (template MAE and beat timing worse).
Fold 9 MAE was the highest of folds 1–9; fold 10 may differ (device mix). Whatever the
outcome, it is reported unchanged; nothing is re-tuned afterwards.
