# Phase 12 pre-registration: robustness

Written 2026-11-24 before any Phase 12 run. Model: the Phase 11 final configuration
(`reports/phase11_tuning/stage_b.json`).

## Data

**Out-of-fold (OOF) forecasts:** 9-fold CV over folds 1–9 (3 seeds per fold, 3-seed
ensemble), so every cleaned development record (19,571) is forecast once by a model
that did not train on its patient. If the final configuration is the Phase 10 base,
the Phase 10 stage-C runs are reused. References: the Phase 9 L1 CNN-LSTM CV (same
folds and seeds), flat and `sn4`.

## Analyses

1. **Strata** (defined from metadata or the history only): device family (AT-6, AT-60,
   CS-12, CS100), quality flag (any PTB-XL quality annotation vs none), baseline
   wander tertile (Phase 3 median peak-to-peak), rhythm (sinus; sinus tachy/brady/
   arrhythmia; AF/flutter; paced; other), heart rate (<60, 60–100, >100 bpm, Phase 2),
   human validation. Per stratum: patient-macro MAE and gatekeepers for all four arms;
   for strata with ≥ 200 patients, the D6.2 rule (`stats.compare`, patient-cluster
   bootstrap on the OOF ensemble) of final vs flat, L1 and `sn4`. Run-to-run variance is
   not included here (one OOF ensemble per record); fold 9 covers it (Phase 10).
2. **Fold stability:** SD of the per-fold MAE.
3. **Input perturbations (fold 9):** 6 fold-9 models of the final configuration are
   retrained with saved weights (their forecasts must equal the existing runs bitwise).
   Histories are perturbed at inference only: Gaussian noise σ 0.02 mV and 0.05 mV
   (seed 0), and lead II set to 0 (detached electrode). `sn4` and flat are recomputed
   from the perturbed history; scoring is against the clean truth. `compare_runs`
   (6 runs) vs perturbed flat and perturbed `sn4`.

## Hypotheses

| ID | Hypothesis |
|---|---|
| H12.1 | Final beats flat in every stratum with ≥ 200 patients |
| H12.2 | Final beats L1 in every stratum with ≥ 200 patients |
| H12.3 | In AF/flutter, final's beat timing is not worse than `sn4`'s (CI of Δ includes or is below 0) |
| H12.4 | Final still beats flat under every perturbation |
