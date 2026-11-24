# Phase 16: final model

*2026-11-24 · pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md) (amendment 1:
hidden 128, D12.1) · frozen before fold 10 was opened*

Reproduce: training via `research/phase16_final.py` specs (`research/run_12_18.py`),
manifest via `uv run --locked python research/phase16_final.py`.

## Frozen models

| | Final | L1 reference |
|---|---|---|
| Model | CNN-LSTM (stem width 64, 2-layer LSTM, hidden 128) with a per-step, per-lead sigmoid gate on the `sn4` forecast + correction head (Phase 10 configuration) | Phase 9 CNN-LSTM (hidden 128), L1 |
| Parameters | 1,168,816 | 719,896 |
| Input | 7 s history, 0.67 Hz causal baseline + residual (24 ch.); `sn4` from the history | same, without `sn4` |
| Output | 2.9 s, 12 leads, raw mV; mean of 6 seeds | same |
| Training data | 15,592 records: cleaned folds 1–9 minus the 20% internal patient holdout | same |
| Seeds | 42, 123, 999, 7, 2024, 31337 | same |
| Best epoch / epochs run | 39–52 / 44–57 | 71–84 / 76–89 |
| Holdout L1 (best) | 0.0863–0.0870 | 0.0883–0.0890 |
| Minutes per run (CPU, 4 threads) | 28–40 | 59–79 |

No run reached the 100-epoch cap. Weights, fitted pipelines, their SHA-256 hashes and
the configuration are in [`MANIFEST.json`](MANIFEST.json) (frozen in commit
`d30951e`); the weight files are git-ignored (`.cache/data/phase16/`).

**History.** Phase 16 first froze the Phase 11 choice (hidden 256). Phase 12's 9-fold
CV showed it worse than hidden 128 on template MAE and timing, so the user reverted to
hidden 128 (D12.1) before fold 10 was opened. The superseded models are documented in
[`MANIFEST_superseded_hidden256.json`](MANIFEST_superseded_hidden256.json). Phases 12,
13, 14, 15 and 18 analysed the hidden-256 configuration; the hidden-128 configuration
was analysed in Phase 10 (CV) and differs by ≈ 1 µV MAE.
