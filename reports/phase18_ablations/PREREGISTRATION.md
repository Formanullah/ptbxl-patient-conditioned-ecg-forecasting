# Phase 18 pre-registration: ablations

Written 2026-11-24 before any Phase 18 run. Fold 9, 6 seeds per arm, `compare_runs`
against the final configuration's 6 fold-9 runs (Δ = ablation − final).

| Ablation | Change | Hypothesis |
|---|---|---|
| no gate | forecast = `sn4` + correction | H18.1: MAE worse (CI > 0): `sn4`'s errors pass through unchecked |
| `sn4` as input only | plain CNN-LSTM, `sn4` as 12 extra input channels, no skip path | H18.2: beat timing worse (CI > 0): the model cannot copy beats through its bottleneck |
| raw input | no 0.67 Hz decomposition (D5.11) | H18.3: MAE difference not significant (D5.11 effect was 0.34 µV) |
| no `sn4` | Phase 9 L1 CNN-LSTM (existing runs) | reported (Phase 10 result) |
| `sn4` alone | the baseline itself | reported |

Ablations on the development data only; they do not change the frozen final model.
