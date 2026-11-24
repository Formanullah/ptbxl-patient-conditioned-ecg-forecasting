# Phase 16 pre-registration: final model

Written 2026-11-24 before any Phase 16 run.

- **Final model:** the Phase 11 final configuration, trained on **all cleaned records of
  folds 1–9** minus the 20% internal patient holdout (seed 0, early stopping), seeds
  42, 123, 999, 7, 2024, 31337. **Prediction = mean of the 6 forecasts.**
- **Reference for the test:** the Phase 9 L1 CNN-LSTM, trained identically (6 seeds).
- **Freeze:** weights, fitted pipelines, configuration, seeds and git commit are written
  to `MANIFEST.json` with SHA-256 hashes. Phase 17 verifies the hashes before loading
  fold 10. No change to the models or the analysis is allowed after the freeze.
- Fold 10 is not touched in this phase.
