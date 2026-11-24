# Phase 18: ablations

*2026-11-24 · pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md) · fold 9, 6 seeds per arm*

Reproduce: `uv run --locked python research/phase18_ablations.py` (after the training queue).

| Arm | Params | MAE single | MAE ens. | DTW-MAE | Template | Timing (ms) | ≤ 50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| **final** | 2.68 M | 0.0882 | 0.0851 | 0.0694 | 0.0934 | 126 | 30% |
| no gate (`sn4` + correction) | 2.68 M | 0.1043 | 0.1017 | **0.0610** | **0.0643** | **89** | **43%** |
| `sn4` as input only | 1.80 M | 0.0929 | 0.0893 | 0.0788 | 0.1290 | 264 | 0.3% |
| raw input (no decomposition) | 2.68 M | **0.0870** | **0.0841** | 0.0684 | 0.0915 | 120 | 31% |
| no `sn4` (L1 CNN-LSTM) | 0.72 M | 0.0911 | 0.0874 | 0.0781 | 0.1288 | 267 | 0.4% |
| `sn4` alone | – | 0.1220 | – | 0.0799 | 0.0642 | 89 | 43% |

`compare_runs`, ablation − final (µV; timing ms), 95% CI:

| Ablation | ΔMAE | ΔDTW | ΔTemplate | ΔTiming | Beats final? |
|---|---|---|---|---|:-:|
| no gate | **+16.1** [+14.6, +17.8] | −8.4 | **−29.1** | **−37.2** | ✗ (MAE) |
| `sn4` as input only | +4.8 [+3.8, +5.9] | +9.4 | +35.5 | +137.6 | ✗ |
| raw input | **−1.1 [−1.8, −0.5]** | −1.0 [−1.8, −0.3] | −1.9 [−4.6, +0.7] | −6.3 [−13.0, +0.5] | **✓** |
| no `sn4` | +2.9 [+2.0, +3.9] | +8.7 | +35.4 | +140.4 | ✗ |
| `sn4` alone | +33.8 [+32.1, +35.5] | +10.5 | −29.2 | −37.3 | – |

| Hypothesis | Result |
|---|---|
| H18.1 removing the gate worsens MAE | **Accepted** (+16 µV) |
| H18.2 `sn4` as input only loses timing | **Accepted** (+138 ms: no better than without `sn4`) |
| H18.3 raw input makes no significant difference | **Rejected: raw input is better** on MAE and DTW |

## Interpretation

- **The skip path is essential.** Given `sn4` only as input, the CNN-LSTM ignores it
  (timing 264 ms, like no `sn4` at all). The timing skill comes from passing `sn4`
  through to the output.
- **The gate trades shape for MAE.** Without it (`sn4` + correction), the model keeps
  full-amplitude beats: template and timing match `sn4`, DTW is the best of all arms,
  but MAE is 16 µV worse (still 18 µV better than `sn4` alone). The gate damps beats
  because, under L1, a damped beat is the safer bet when its timing is uncertain.
- **The 0.67 Hz decomposition adopted in D5.11 (on the reference CNN) hurts this
  model:** raw input is better by 1.1 µV MAE, with no gatekeeper worse. It is a single
  fold-9 comparison; it was not re-checked with 9-fold CV.
- Ablations do not change the frozen model. Whether to act on them is D18.1 (user).
