# Leakage audit

The executable audit is `python src/pipeline.py audit`. Read the current study's
`leakage_audit.json` for executed checks; this document describes the controls.

- Reject duplicate ECG IDs, missing IDs/folds and patients occurring in multiple official folds.
- Select whole patients, keeping every eligible recording, with independent seeded streams for training and validation.
- Use one start-zero 700/300 window per recording. No sliding overlap or unused tail segment exists.
- Fit waveform mean/std on raw training history only. P0/P1 share this scaler and unfiltered targets.
- Fit age/sex and expanded-feature transformations on training records only.
- Filter observed samples only; mutating future targets cannot change inputs or observed-only HR/RR features.
- Verify causal prefix invariance when later historical samples change.
- Exclude current/equal-time annotations from prior history; unavailable chronology yields a missing flag.
- Fit mRMR bins/rankings on training features and training-target summaries only. Grouped permutation may use validation, never test.
- Select checkpoints and recipes using validation patient-macro MAE in original mV.
- Require a frozen four-condition selection and explicit final-test opt-in before preparing/loading fold-10 signals.
- Verify locked checkpoint/configuration/validation/scaler hashes before test inference. Reuse completed ledger entries; never select a different model based on test outcomes.
- Reject stale cache files, metadata changes, source-waveform changes, legacy protocol schemas and incompatible resumes.
- Keep synthetic and pilot diagnostics out of research rankings. Mark missing supporting artifacts unavailable.

The original suite passed 60 checks before refactoring. The new suite tests these
contracts directly, including resume equivalence, paired patient metrics and the
final ledger. Test output and actual pilot execution are reported separately from
any unfinished GPU research experiments. A failed check stops the applicable stage.
