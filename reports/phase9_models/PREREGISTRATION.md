# Phase 9 pre-registration: model families under a fixed L1 objective

Written 2026-11-22 before any Phase 9 training run. The only measurements taken
beforehand were CPU training speeds on random data, used to size the models.

Scope (user decision, 2026-11-22): this phase compares **architectures**. Everything
else is held fixed: the L1 loss, the data and the training protocol.
Timing/shape-aware objectives are deferred to Phase 10, which carries forward the top
two families.

## Fixed setup

| Item | Setting | Source |
|---|---|---|
| Data | cleaned folds 1–8 minus the 20% internal patient holdout (seed 0); full pool | D3.1, D3.4, D7.2 |
| Input | 0.67 Hz causal baseline + residual, 24 channels; raw-mV target, 12 leads | D5.11, D5.5 |
| Training | L1 loss, AdamW (weight decay 1e-4), batch 64, up to 40 epochs, early stopping with patience 5 on the internal holdout, gradient clipping 1.0, 4 threads, deterministic | Phases 5–8 |
| Evaluation | fold 9 (screening); 9-fold CV over folds 1–9 for the shortlist; Phase 6 suite | D8.1, D8.2, D8.6 |
| Comparison | `stats.compare_runs` (6 vs 6 independently seeded runs; bootstrap over runs, then patients) for MAE **and** gatekeepers. Deterministic references (flat, `sn4`) enter as a single run | D8.3 |

## Families (sized to similar CPU cost; parameters and speed measured on random data)

| Family | Configuration | Parameters | s/epoch |
|---|---|---:|---:|
| CNN (reference) | 3 strided conv blocks, width 64, linear head | 1.85 M | 6 |
| LSTM | 1 layer, hidden 64, on all 700 steps; head from the last hidden state | 0.25 M | 44 |
| CNN-LSTM | 2 strided conv blocks (width 64, 700 → 175 steps) + 2-layer LSTM, hidden 128 | 0.72 M | 55 |
| PatchTST | channel-independent per lead; 100 ms patches (20 samples, stride 20) holding the lead's 2 channels; RevIN; d 64, 4 heads, 2 layers, feed-forward 128; flatten head | 0.74 M | 89 |
| Transformer | channel-mixing encoder-decoder: 70 tokens of 100 ms × 24 channels; d 128, 8 heads, 3 encoder + 2 decoder layers; 29 learned horizon queries, each producing 100 ms × 12 leads | 1.18 M | 82 |

PatchTST deviates from the published setting (patch 16, stride 8, d 128, 3 layers): that
costs about 1,000 s/epoch, about 6 h per run on this CPU. The reduction is a
compute-driven choice and is disclosed.

## Stages

**A — Learning rate (internal holdout only).** Each family, including the CNN, is
trained with seed 42 at learning rates 1e-3 and 3e-4. The rate with the lower best
internal-holdout L1 is kept. Fold 9 is not used.

**B — Fold-9 screening.** Each family trains at its chosen rate with seeds 42, 123,
999, 7, 2024 and 31337. Comparisons, using `compare_runs` and the full D6.2 logic:
- every family vs the CNN reference;
- every family vs flat and vs `sn4`.

Reported per family:
- single-model seed-mean MAE and its SD;
- 6-run ensemble metrics, including the share of records with beat timing ≤ 50 ms;
- epochs and wall-clock time.

**C — Multi-fold confirmation (D8.6).**
- **Shortlist:** the two new families with the lowest fold-9 single-model seed-mean
  MAE.
- **Evaluation:** each shortlisted family is trained for every held-out official fold
  k = 1…9 (training on the other eight folds minus the 20% internal holdout), with
  seeds 42, 123 and 999.
- **CNN reference:** its runs come from Phase 8 (same protocol).
- **Per fold:** 3-seed ensemble, patient-macro suite.
- **Across folds:** paired t-tests on the 9 per-fold differences.

**D — 8-lead output.** The family with the lowest fold-9 seed-mean MAE (CNN included)
is retrained with 8-lead output: I, II, V1–V6 predicted; III, aVR, aVL and aVF derived
exactly; 16 input channels. Same 6 seeds, compared with its 12-lead version using
`compare_runs`. Limb-lead consistency is reported as in Phase 5 (worst RMS violation
per record).

## Hypotheses

- **H9.1 — Architecture does not fix rhythm under L1 (generalizes D5.9).** No family
  beats `sn4` under the comparison rule, and every family's 6-run ensemble has a
  patient-macro beat-timing error above 150 ms.
- **H9.2 — Architecture has a small effect on MAE.** The best family's fold-9
  seed-mean MAE is within 2% of the CNN reference's.
- **H9.3 — Fold 9 ranks the shortlist as CV does (consistent with H8.2).** Among the
  CV families (two shortlisted + CNN), the fold-9 MAE ranking of 3-seed ensembles
  equals the 9-fold mean ranking.
- **H9.4 — 8-lead output is non-inferior.** For the stage-D family, the ΔMAE (8 − 12
  leads) CI upper bound is ≤ +0.5 µV, and no gatekeeper's CI is entirely worse.

## Decision rules

- **Best family under L1 (D8.6).** Family X is declared better than family Y only if
  both hold:
  - on fold 9, X beats Y under `compare_runs` (6 vs 6);
  - across the 9 CV folds, the mean ΔMAE is negative with paired t-test p < 0.05, and
    no gatekeeper is significantly worse (paired t-test p < 0.05 with a positive mean).
  The **best family** beats every other CV family under this rule. If none does, no
  winner is declared under L1.
- **Carried into Phase 10:** the two CV families with the lowest 9-fold mean MAE. A
  final winner is only declared after Phase 10, again with multi-fold confirmation
  (D8.6).
- **8-lead output** is adopted if H9.4 holds, because it guarantees lead consistency;
  otherwise 12-lead output is kept.

## Limitations fixed in advance

- **One configuration per family, with only the learning rate checked.** This
  compares families at a similar cost, not each family at its best. Phase 11
  (hyperparameters) can revisit the shortlisted families.
- **Similar CPU cost does not mean similar capacity.** Parameter counts differ by up
  to 7×.
- **9-fold CV uses 3 seeds per fold, as in Phase 8.** Seed noise is part of the
  between-fold variance, so the across-fold t-tests are conservative.

## Amendment 1 (2026-11-23, after stage A; user decision)

**Change:** the maximum number of epochs is raised from 40 to **100** for every
family. Early stopping (patience 5 on the internal holdout) is unchanged.

**Why:** in stage A, the 40-epoch cap was binding. At their better learning rate,
LSTM (best epoch 37 of 40), CNN-LSTM (39 of 40) and the Transformer at 3e-4 (40 of 40)
were still improving when training stopped. CNN and PatchTST stopped early (27 and 34
epochs). At 40 epochs the comparison would partly measure convergence speed instead of
architecture.

**Reproducibility of the change:** runs that early-stopped before epoch 40 are
bit-for-bit identical under the new cap, so they are kept. The 6 capped stage-A runs
are retrained and the learning-rate choice is re-made with the same rule. The capped
results are kept in `table_stage_a_lr_cap40.csv`.

**Bias check:** the change was decided before any fold-9 result existed. It applies to
all families equally. The decision rules are unchanged.
