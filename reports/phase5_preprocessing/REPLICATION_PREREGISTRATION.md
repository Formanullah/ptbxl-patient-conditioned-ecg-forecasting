# Phase 5 addendum: replication test (pre-specified)

Written 2026-11-22, after the seeded re-run of the Phase 5 CNN tier and **before** any
replication run.

## Why

Before the initialization fix (D7.1), `train_forecaster` did not seed the initial
weights. The original Phase 5 CNN runs therefore used random, unrecorded initializations.
The seeded re-run changed two verdicts under the D6.2 rule:

| Variant | Unseeded run, ΔMAE vs default (µV) [CI] | Seeded run (42/123/999) | D6.2 verdict |
|---|---|---|---|
| Gaussian-noise augmentation | +0.31 [+0.08, +0.71] | −0.18 [−0.27, −0.09] | ✗ → ✓ |
| Decomposition 0.67 Hz | +0.01 [−0.26, +0.36] | −0.37 [−0.84, −0.03] | ✗ → ✓ |

The D6.2 bootstrap resamples patients only. It does not capture variation between
training runs.

## Test

- **Runs.** Default, `aug_gaussian_noise` and `filter_decompose_0.67`; new seeds 7,
  2024 and 31337; otherwise identical to Phase 5.
- **Replication criterion.** A candidate replicates if its 3-seed ensemble for the new
  seeds again beats the new-seed default ensemble under D6.2.
- **Adoption.** Adopt only if it replicates. In addition, the 6-seed ensemble
  (42/123/999/7/2024/31337) must beat the 6-seed default under D6.2.
- Otherwise the variant is recorded as "not robust to training randomness" and is not
  adopted.
