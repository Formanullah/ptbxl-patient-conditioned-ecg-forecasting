# Phase 17: one-time test on fold 10

*Run once on 2026-11-24 at 22:12 (lock file `UNSEALED.lock`) · pre-registered in
[PREREGISTRATION.md](PREREGISTRATION.md) (amendment 1: hidden-128 final model) ·
models frozen at commit `d30951e` ([Phase 16 manifest](../phase16_final/MANIFEST.json),
hashes verified before loading fold 10)*

Command: `uv run --locked python research/phase17_test.py --unseal-fold-10-once`
(refuses to run again).

**Test set:** fold 10 after the same cleaning rules: 2,198 records, 1,904 patients,
none of whom appear in folds 1–9 (official patient-level split).

## Results

| Arm | MAE single (6-seed mean) | MAE 6-run ens. | DTW-MAE | Template MAE | Timing (ms) | ≤ 50 ms |
|---|---:|---:|---:|---:|---:|---:|
| **Final** (`sn4`-gated CNN-LSTM) | 0.0883 | **0.0850** | **0.0682** | 0.0902 | 110 | 36% |
| L1 CNN-LSTM | 0.0900 | 0.0861 | 0.0749 | 0.1363 | 270 | 0.2% |
| Flat | – | 0.1207 | 0.1207 | 0.1055 | 299 | 0.3% |
| `sn4` | – | 0.1239 | 0.0819 | **0.0660** | **88** | **44%** |

Ensemble columns are the 6-run ensembles (patient-macro).

**Primary analysis** (`compare_runs`, final's 6 runs − reference, µV / ms, 95% CI):

| vs | ΔMAE | ΔDTW | ΔTemplate | ΔTiming | Verdict |
|---|---|---|---|---|:-:|
| flat | −32.4 [−34.4, −30.7] | −50.5 | −12.6 [−15.5, −9.5] | −179 | **beats** |
| L1 CNN-LSTM | −1.7 [−2.6, −0.9] | −5.9 | −40.6 [−43.9, −37.4] | −145 | **beats** |
| `sn4` | −35.6 [−37.2, −34.2] | −11.7 | **+27.0** [+23.5, +30.4] | **+31.6** [+26.2, +37.6] | does not beat |

## Against the pre-registered expectations

| Expectation (from 9-fold CV) | Fold 10 |
|---|---|
| MAE ≈ 0.085 mV | 0.0850 (ensemble) ✓ |
| template ≈ 0.088 mV | 0.0902 |
| timing ≈ 109 ms | 110 ✓ |
| beats flat and L1 | ✓ both |
| does not beat `sn4` on template and timing | ✓ (worse on both) |

**The development results generalise to the held-out patients.** The final model is
more accurate than every baseline pointwise and shape-tolerantly, and better than the
L1 CNN-LSTM on every metric. It is still less faithful than simply repeating the last
beats on beat morphology and timing.

## Notes

- This is the only fold-10 evaluation; nothing was changed after it.
- The MAE gain over L1 (1.7 µV) matches the 9-fold CV (1.6 µV) and is small in absolute
  terms (2% of MAE); the gatekeeper gains over L1 are large.
- The dry run on fold 9 before unsealing (in-sample, not a result) checked the code
  path; its outputs are not reported.
