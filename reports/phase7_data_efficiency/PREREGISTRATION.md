# Phase 7 pre-registration: data efficiency

Written 2026-11-22 before any Phase 7 computation.

## Design

- **Pool.** Cleaned folds 1–8 minus the fixed internal patient holdout (20% of
  patients, seed 0, the same as Phase 5). The holdout is used only for early stopping
  and is never subsampled.
- **Fractions.** 5, 10, 20, 40, 60, 80 and 100% of the pool's **patients** (all their
  records), as nested subsets: for each seed, one random patient order, and each
  fraction takes a prefix of it.
- **Seeds.** 42, 123, 999. Each seed fixes both the subset order and the model
  initialization.
- **Learners.**
  - Phase 5 reference CNN, default pipeline (raw mV), L1 loss, AdamW at 1e-3, batch
    64, up to 40 epochs, patience 5, 4 threads per run.
  - Ridge reference (λ from its own internal holdout), as the linear comparison.
- **Evaluation.** Fold 9 with the Phase 6 suite: patient-macro MAE (primary),
  DTW-MAE, template MAE, beat timing and the share of records ≤ 50 ms. Curves show the
  mean ± SD over seeds; comparisons between fractions use 3-seed ensembles and the
  patient-cluster bootstrap.
- **Cost.** Wall-clock training time, epochs to the best holdout loss, and records
  processed.

## Definitions (fixed now)

- **Minimum useful size:** the smallest fraction whose CNN ensemble beats the flat
  baseline under the D6.2 rule.
- **Saturation point:** the smallest fraction whose ensemble MAE is within 1% of the
  100% ensemble **and** whose ΔMAE CI includes 0.
- **Diminishing returns:** the first doubling step (5→10, 10→20, 20→40, 40→80% of
  patients) that improves ensemble MAE by less than 1%.

## Hypotheses

- **H7.1 — The CNN's MAE saturates before the full pool.** The 60% ensemble's MAE is
  within 1% of the 100% ensemble's MAE.
- **H7.2 — More data does not fix rhythm under an L1 objective (tests D5.9).** Going
  from 20% to 100% improves the ensemble beat-timing error by less than 10%, and the
  100% CNN still does not beat `sn4` on beat timing.
- **H7.3 — Small data overfits more.** The generalization gap at 5% is at least twice
  the gap at 100% (seed means). The gap is the holdout L1 minus the training L1, both
  at the selected (best) epoch, i.e. the model that is actually evaluated.

## Reproducibility check

The 100% runs repeat Phase 5's default runs exactly (same pool, holdout, seeds and
thread count). Their fold 9 predictions must be **bitwise identical** to the Phase 5
cache.
