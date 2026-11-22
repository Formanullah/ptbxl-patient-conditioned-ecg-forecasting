# 100 Hz implementation plan

The existing 500 Hz pilot checkpoints, tables, figures, report and presentation files remain historical artifacts. They are not directly comparable to 100 Hz forecasts with a longer horizon.

1. Reuse `metadata`, strict chronological `static_features`, WFDB checks, cached Dataset/DataLoader interfaces and protocol fingerprints. Add a versioned 100 Hz preparation path with configurable seconds, patient cohorts, training sliding windows, raw-mV targets and train-only transforms. Validation/test use one start-zero window per recording.
2. Extend existing `Forecaster` and training/checkpoint functions with optional early fusion, configurable CNN geometry, losses, schedulers, complete epoch budgets and validation patient-macro MAE selection. Preserve the old defaults and checkpoint keys for legacy runs.
3. Reuse voltage metrics, plots and patient bootstrap. Add explicit validation evaluation, guarded final testing, one-to-one R-peak matching and morphology plots. Never use fold 10 in selection.
4. Add explicit manual scripts for preprocessing diagnostics, training-only mRMR ranking and grouped validation permutation ranking. Keep all twelve waveform leads. Save selected feature names and their fitting provenance.
5. Test data, record, patient and temporal isolation before optimization. Run real-data development comparisons on a clearly identified custom CPU cohort. Compare preprocessing, features, fusion, learning rates, model dimensions and complete 60/100-epoch budgets as feasible.
6. Generate measured validation tables and research documentation. Supply runnable YAMLs/commands for larger cohorts and official final evaluation. Record failed and pending experiments rather than manufacturing results.

Primary selection objective: lowest validation patient-macro MAE in original mV. Secondary criteria: RMSE, peak metrics, stability, runtime and complexity. Final fold-10 evaluation is an explicit separate action after a validation-based selection record.
