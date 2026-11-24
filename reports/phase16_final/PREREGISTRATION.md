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

## Amendment 1 (2026-11-24, before fold 10 was opened)

User decision D12.1: the final model reverts to the **Phase 10 configuration (LSTM
hidden 128, 0.67 Hz decomposition)**, because Phase 12's 9-fold CV found hidden 256
worse on template MAE (+1.3 µV, p = 0.013) and beat timing (+5.2 ms, p = 0.002), and
the multi-fold criterion (D8.6) favours hidden 128. D18.1: the decomposition is kept
(raw input was better on fold 9 only, not CV-confirmed). The 6 final models are
retrained with hidden 128 and re-frozen; the hidden-256 models are kept as superseded
(`MANIFEST_superseded_hidden256.json`). The L1 reference is unchanged.
