# Phase 10 pre-registration: training objectives (CNN-LSTM)

Written 2026-11-23 before any Phase 10 training run. The only computation done
beforehand was caching the `sn4` forecasts and checking them against Phase 6
(identical up to float32 rounding: max |ΔMAE| 7×10⁻⁸ mV, `sn4_check.json`).

## Question

Under L1, every family forecasts a blurred average (Phase 9): CNN-LSTM has the lowest
MAE but a worse beat morphology than a flat line (template MAE +24 µV on fold 9) and
no beat timing skill (~267 ms vs `sn4` 89 ms). Does a timing/shape-aware objective fix
this without losing the MAE gain?

## Scope (user decisions, 2026-11-23)

- **Objectives:** derivative loss, multi-resolution spectral loss and `sn4`
  conditioning. Soft-DTW is deferred until a GPU is available (≈10–30× slower on CPU).
- **Family: CNN-LSTM only.** *Deviation from D9.4*, which carried CNN-LSTM and the
  Transformer. The Transformer costs about twice as much per run (110 vs 53 min) and is
  dropped to halve the compute. Phase 10 results therefore apply to CNN-LSTM only.

## Fixed setup (as Phase 9)

Data, splits, input (0.67 Hz decomposition, 24 channels), raw-mV 12-lead target,
AdamW (lr 1e-3, D9.2; weight decay 1e-4), batch 64, up to 100 epochs (D9.1), early
stopping with patience 5 **on the holdout value of the run's own objective**, gradient
clipping 1.0, 4 threads, deterministic. Evaluation with the Phase 6 suite; comparisons
with `stats.compare_runs` (D8.3) on MAE **and** gatekeepers.

**L1 reference:** the cached Phase 9 CNN-LSTM runs (same seeds, splits and settings).
The training loop gains an optional `loss` argument; a rerun of the L1 run (fold 9,
seed 42) must be bitwise identical to the cached Phase 9 run, otherwise the reference
is retrained.

## Objectives

| Name | Loss | Notes |
|---|---|---|
| `l1` | L1 | Phase 9 reference |
| `derivative` | L1 + λ·c·L1(Δŷ, Δy), Δ = first difference in time | rewards correct slopes (QRS edges) |
| `spectral` | L1 + λ·c·mean over n_fft ∈ {16, 32, 64} of L1 between STFT magnitudes (Hann, hop n_fft/4) | tolerant to small time shifts, penalises missing or blurred waves |
| `sn4_gated` | L1, model `GatedBaselineCNNLSTM`: ŷ = σ(g(h)) ⊙ sn4 + r(h) | `sn4` forecast (history only) as input; per-step, per-lead gate; closed gate = Phase 9 CNN-LSTM, open gate + r = 0 = `sn4`. Adds a second linear head (+0.45 M parameters, 1.17 M in total) |

`c` normalises the auxiliary term: L1 / aux of the flat (history-median) forecast on
every 7th training record of the split, so λ = 1 means "as large as L1 for a flat
forecast". `sn4` is computed from each record's history only (causal, no leakage).

## Stages

**A — Loss weight (internal holdout only; fold 9 unused).** Seed 42, λ ∈ {0.3, 1, 3}
for `derivative` and `spectral`. Each run is scored with the Phase 6 suite on the 20%
internal patient holdout (patient-macro). **Rule:** among the λ whose holdout MAE is
≤ 1.02 × the L1 run's holdout MAE, keep the one with the lowest holdout template MAE;
if none qualifies, keep the λ with the lowest holdout MAE. `sn4_gated` has no weight.
The L1 rerun (bitwise check) is trained in this stage.

**B — Fold-9 screening.** Each objective at its chosen λ, 6 seeds (Phase 9 seeds;
the stage-A seed-42 run is reused). `compare_runs` of each objective against the L1
CNN-LSTM, flat and `sn4`.

- **Eligible:** beats flat under D6.2/D8.3 (ΔMAE CI < 0 and no gatekeeper CI > 0).
  The L1 CNN-LSTM is not eligible (template MAE worse than flat).
- **Finalists:** the eligible objectives, at most two, by lowest single-model mean MAE.
  If none is eligible, there is no stage C and the phase reports a negative result.

**C — Multi-fold confirmation (D8.6).** Finalists trained on 9-fold CV (folds 1–9,
seeds 42/123/999, 3-seed ensemble per fold), compared with flat, `sn4` and the Phase 9
L1 CNN-LSTM CV runs. Per-fold differences, one-sample t-test across folds. An arm
**beats** a reference if mean ΔMAE < 0 with p < 0.05 and no gatekeeper is worse with
p < 0.05. An objective is **adopted** only if it beats flat on fold 9 (stage B) and in
the CV. Overlapping training sets make the CV p-values approximate (Phase 9).

## Hypotheses (predictions written before the runs)

| ID | Hypothesis | Test |
|---|---|---|
| H10.1 | `derivative` and `spectral` lower template MAE vs L1 | stage B, template ΔCI < 0 vs `l1` |
| H10.2 | `sn4_gated` is eligible (beats flat) and its beat timing is better than L1's | stage B: beats flat; timing ΔCI < 0 vs `l1` |
| H10.3 | No objective beats `sn4` on every gatekeeper | stage B, `compare_runs` vs `sn4` |

## Reporting

Single-model (seed mean) and 6-run ensemble results separately; share of records with
beat timing ≤ 50 ms; runs that hit the 100-epoch cap; per-run minutes. Fold 10 stays
sealed.

## Cost (CPU, 10 workers × 4 threads)

CNN-LSTM ≈ 53 min per run under L1. Stage A + `sn4_gated` seeds: 13 runs; stage B:
10 more; stage C: 27 runs per finalist. Expected ≈ 3–4 h to the end of stage B,
≈ 3–7 h for stage C.
