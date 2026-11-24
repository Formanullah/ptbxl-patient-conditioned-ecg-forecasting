# Phase 16: final model

*2026-11-24 · pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md) · fold 10 not touched*

Reproduce: `uv run --locked python research/run_12_18.py` (training) and
`uv run --locked python research/phase16_final.py` (manifest).

## Frozen models

| | Final | L1 reference |
|---|---|---|
| Model | CNN-LSTM (stem width 64, 2-layer LSTM, hidden 256) with a per-step, per-lead sigmoid gate on the `sn4` forecast + correction head | Phase 9 CNN-LSTM (hidden 128), L1 |
| Parameters | 2,684,336 | 719,896 |
| Input | 7 s history, 0.67 Hz causal baseline + residual (24 ch.); `sn4` from the history | same, without `sn4` |
| Output | 2.9 s, 12 leads, raw mV; mean of 6 seeds | same |
| Training data | 15,592 records: cleaned folds 1–9 minus the 20% internal patient holdout | same |
| Seeds | 42, 123, 999, 7, 2024, 31337 | same |
| Best epoch / epochs run | 25–33 / 30–38 | 71–84 / 76–89 |
| Holdout L1 (best) | 0.0855–0.0863 | 0.0883–0.0890 |
| Minutes per run (CPU, 4 threads, under load) | 76–95 | 59–79 |

No run reached the 100-epoch cap. Weights, fitted pipelines and their SHA-256 hashes,
the configuration and the freezing commit (`ac60f77`) are in
[`MANIFEST.json`](MANIFEST.json); the weight files themselves are git-ignored
(`.cache/data/phase16/`).

**Status: frozen, pending D12.1.** Phase 12 found that the selected hidden size (256)
costs template MAE and timing across folds compared with hidden 128. If the user
decides to revert, Phase 16 is retrained with hidden 128 and re-frozen *before* fold 10
is opened. Phase 17's dry run on fold 9 (in-sample, numbers meaningless) checked that
the frozen files load, hash-verify and score correctly.
