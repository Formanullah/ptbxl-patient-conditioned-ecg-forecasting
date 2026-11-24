# Phase 11: hyperparameters (sn4-gated CNN-LSTM)

*2026-11-24 · pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md) · fold 10 sealed*

```sh
uv run --locked python research/phase11_tuning.py --stage A
uv run --locked python research/phase11_tuning.py --stage B
```

## Summary

- **Final configuration: LSTM hidden size 256** (otherwise as Phase 10: width 64,
  2 layers, lr 1e-3, batch 64, L1). On fold 9 (6 vs 6 runs) it beats the Phase 10 base
  on MAE by **−0.77 µV [−1.44, −0.18]** (0.9%), with no gatekeeper worse.
- The gain is small and costs 2.3× the parameters (2.68 M) and about 2× the training
  time. It is adopted because it meets the pre-registered rule; it is not a large
  improvement. **H11.1 (no candidate beats base) is rejected.**
- A seed-42 rerun of the base reproduced Phase 10 **bitwise**.

## Stage A: one change at a time (internal holdout, seed 42)

| Candidate | Params | MAE | DTW-MAE | Template MAE | Timing (ms) | Best epoch | Min |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | 1.17 M | 0.08652 | 0.0691 | 0.0912 | 119 | 45 | 45 |
| `lr3e-4` | 1.17 M | 0.09194 | 0.0733 | 0.0851 | 91 | 37 | 30 |
| **`hidden256`** | 2.68 M | **0.08553** | 0.0668 | 0.0913 | 124 | 33 | 88 |
| `hidden64` | 0.56 M | 0.09051 | 0.0743 | 0.0867 | 96 | 47 | 24 |
| `layers3` | 1.30 M | 0.08719 | 0.0699 | 0.0986 | 140 | 47 | 61 |
| `width128` | 1.30 M | 0.08766 | 0.0691 | 0.0876 | 113 | 38 | 37 |
| `spectral0.1` | 1.17 M | 0.08787 | 0.0666 | 0.0840 | 109 | 48 | 45 |
| `spectral0.3` | 1.17 M | 0.09015 | **0.0630** | **0.0745** | 95 | 49 | 44 |

Only `hidden256` improved holdout MAE by ≥ 0.5 µV (−0.99 µV), so it is the only
shortlisted candidate.

**Trade-off visible in this grid (single seed, descriptive):** every candidate that
lowers MAE less (smaller model, lower learning rate, spectral term) forecasts sharper,
better-timed beats. The spectral term at λ 0.3 gives the best template MAE (0.0745 vs
0.0912) and timing (95 vs 119 ms) at +3.6 µV MAE. The project rule (MAE primary,
gatekeepers non-inferior) selects for MAE; a rule weighting morphology would pick
differently.

## Stage B: fold-9 confirmation (6 seeds)

| | MAE single | MAE 6-run ens. | DTW-MAE | Template MAE | Timing (ms) | ≤ 50 ms (ens.) |
|---|---:|---:|---:|---:|---:|---:|
| `base` (Phase 10) | 0.08895 | 0.08557 | 0.0710 | 0.09342 | 125 | 35.1% |
| `hidden256` | **0.08818** | **0.08505** | **0.0694** | 0.09342 | 126 | 33.9% |

`compare_runs` (`hidden256` − `base`, µV / ms, 95% CI): MAE −0.77 [−1.44, −0.18];
DTW −1.55 [−2.39, −0.73]; template −0.00 [−2.24, +2.15]; timing +1.7 [−4.4, +6.9].
**Beats `base`.** (The identical template means are a coincidence: per-record values
differ in 2,108 of 2,179 records; the means differ in the sixth decimal.)

## Limitations

- One change at a time; no interactions (e.g. hidden 256 + spectral term) tested.
- Stage A used one seed; candidates within ~0.5 µV of base were not followed up.
- The selection is on folds 1–9; Phase 12 re-checks the configuration with 9-fold CV.
