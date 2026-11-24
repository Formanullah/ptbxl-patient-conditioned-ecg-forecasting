# Phase 15 pre-registration: error analysis by signal properties

Written 2026-11-24 before any Phase 15 analysis. Data: Phase 12 OOF forecasts.

- **Per lead** and **per horizon** (100 ms bins) MAE for final, L1, `sn4`, flat.
- **History periodicity:** quintiles of the ACF peak height of the `sn4` period estimate
  (history only); MAE and timing per quintile. Expected: final's gain over L1 grows with
  periodicity (the gate should trust `sn4` more when the rhythm is regular).
- **Gate behaviour (fold 9, 6 saved models):** mean gate per horizon step and per lead;
  Spearman correlation of the record-mean gate with periodicity and with `sn4`'s own
  MAE. Expected: positive with periodicity, negative with `sn4` MAE.
- **Worst 1% of records** (final's MAE): over-representation of quality flags, AF,
  pacing, high baseline wander, devices; periodicity and true amplitude.

Exploratory; no decision depends on it.
