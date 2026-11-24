# Phase 11 pre-registration: hyperparameters (sn4-gated CNN-LSTM)

Written 2026-11-24 before any Phase 11 run. Model: the Phase 10 winner, CNN-LSTM with a
learned gate on the `sn4` forecast (D10.2). Everything not listed is as Phase 10.

## Candidates (one change at a time from the base)

| Name | Change |
|---|---|
| `base` | width 64, LSTM hidden 128, 2 layers, lr 1e-3, batch 64, L1 (Phase 10) |
| `lr3e-4` | learning rate 3e-4 |
| `hidden256` / `hidden64` | LSTM hidden size 256 / 64 |
| `layers3` | 3 LSTM layers |
| `width128` | convolutional stem width 128 |
| `spectral0.1` / `spectral0.3` | L1 + λ × spectral term (Phase 10 definition), λ 0.1 / 0.3 |

## Stages

**A — Screening on the internal holdout (seed 42; fold 9 unused).** Each candidate is
scored with the Phase 6 suite on the 20% internal patient holdout. **Shortlist:** up to
two candidates with the lowest holdout MAE among those improving on `base` by
≥ 0.5 µV (about one single-seed SD on fold 9). If none qualifies, `base` is final and
stage B is skipped. A seed-42 rerun of `base` must reproduce the Phase 10 run bitwise.

**B — Fold-9 confirmation.** Shortlisted candidates trained with the 6 Phase 9 seeds;
`compare_runs` against the 6 `base` runs of Phase 10. A candidate **wins** if it beats
`base` under D6.2/D8.3 (ΔMAE CI < 0, no gatekeeper CI > 0). **Final configuration:**
the winner with the lowest single-model MAE, else `base`.

Changes are not combined (no interaction search). The spectral candidates can only be
adopted if they also improve MAE, by the project rule.

## Hypothesis

H11.1: no candidate beats `base` on fold 9 (the CPU-sized base is near a local optimum
for this data size; Phase 7 found diminishing returns).

## Cost

8 runs (one round on 10 workers, ≈ 1–2 h; hidden 256 is the slowest), then ≤ 10 runs
for stage B (≈ 1 h).
