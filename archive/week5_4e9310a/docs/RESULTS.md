# PTB-XL research results

**REAL-DATA SOFTWARE PILOT — numerical diagnostics are excluded from research rankings.**

## Dataset

PTB-XL v1.0.3; source: https://physionet.org/content/ptb-xl/1.0.3/. Data root: `PTB-XL/records100`. WFDB reads physical mV, 100 Hz, twelve leads, 1,000 samples per record. History/target: `700/300`; batched input `[B,700,12]`, output `[B,300,12]`. This is within-record waveform forecasting, not clinical longitudinal prediction.

## Patient-safe folds

Official folds 1–8 train, 9 validate, 10 test; patients remain disjoint. Selected whole patients retain all eligible ECGs.

| split | records | patients | windows |
| --- | --- | --- | --- |
| train | 339 | 300 | 339 |
| val | 81 | 75 | 81 |

Reserved fold 10 metadata counts: {'records': 2198, 'patients': 1904}

Final-test ledger: pending / not enabled

## Preprocessing

P0 uses raw history; P1 uses causal 0.5–40 Hz SOS Butterworth filtering on observed history only (prototype N=4; bandpass order 8). Both use identical raw-training-history z-scores and predict the same original-mV future. Filtering may distort ST segments, QRS shape, phase, and initial samples; it is a hypothesis. History-only distortion/PSD figures do not establish physiological accuracy.

Measured validation filter effects:

Pending: no completed comparable research measurements.

## LSTM

Independent `LSTMForecaster`: optional early static embedding → stacked LSTM → final hidden state → MLP → all future samples. Exact active parameter counts and model settings are in each run.json; see docs/EQUATIONS.md.

| model | filter | features | static_values | hidden | layers | active_parameters | output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | P0 | F0 | 0 | 64 | 1 | 492688 | [B,300,12] |
| lstm | P0 | F1 | 4 | 64 | 1 | 496864 | [B,300,12] |
| lstm | P1 | F0 | 0 | 64 | 1 | 492688 | [B,300,12] |

## CNN-LSTM

Independent `CNNLSTMForecaster`: optional static channels → Conv1d(7,32) → ReLU → Conv1d(5,64) → ReLU → AvgPool(2) → LSTM → MLP. The primary 700-step history becomes 350 recurrent steps. No attention or contrastive projector is active.

| model | filter | features | static_values | hidden | layers | active_parameters | output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm | P0 | F0 | 0 | 64 | 1 | 519024 | [B,300,12] |
| cnn_lstm | P0 | F1 | 4 | 64 | 1 | 522688 | [B,300,12] |
| cnn_lstm | P1 | F0 | 0 | 64 | 1 | 519024 | [B,300,12] |

## Training

AdamW, normalized MSE, fixed learning rates, gradient clipping, seed 42; validation patient-macro MAE selects checkpoints. No early stopping truncates the budget.

| name | state | epochs_completed | best_epoch | runtime_seconds |
| --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.0005_h64_e2 | software smoke check passed; excluded from research comparison | 2 | 2 | 2.96329 |
| cnn_lstm_P0_F1_lr0.0005_h64_e2 | software smoke check passed; excluded from research comparison | 2 | 2 | 3.57382 |
| cnn_lstm_P1_F0_lr0.0005_h64_e2 | software smoke check passed; excluded from research comparison | 2 | 2 | 5.18507 |
| lstm_P0_F0_lr0.0005_h64_e2 | software smoke check passed; excluded from research comparison | 2 | 2 | 4.01756 |
| lstm_P0_F1_lr0.0005_h64_e2 | software smoke check passed; excluded from research comparison | 2 | 2 | 3.55919 |
| lstm_P1_F0_lr0.0005_h64_e2 | software smoke check passed; excluded from research comparison | 2 | 2 | 3.36902 |
| persistence | software smoke check passed; excluded from research comparison | pending | pending | pending |
| repeat_window | software smoke check passed; excluded from research comparison | pending | pending | pending |

Recorded environments: `[{'python': '3.12.9', 'torch': '2.15.0+cpu', 'device': 'cpu', 'gpu': None}, {'python': '3.12.9', 'torch': '2.15.0+cpu', 'device': 'cpu', 'gpu': None}, {'python': '3.12.9', 'torch': '2.15.0+cpu', 'device': 'cpu', 'gpu': None}, {'python': '3.12.9', 'torch': '2.15.0+cpu', 'device': 'cpu', 'gpu': None}, {'python': '3.12.9', 'torch': '2.15.0+cpu', 'device': 'cpu', 'gpu': None}, {'python': '3.12.9', 'torch': '2.15.0+cpu', 'device': 'cpu', 'gpu': None}]`. Full 120-epoch Colab runs remain pending unless their completed artifacts are present.

60/120 checkpoint evidence:

Pending: no completed comparable research measurements.

## RMSE/MAE

MAE = mean(|prediction − truth|), in mV; RMSE = sqrt(mean((prediction − truth)²)), in mV; MSE is mV². RMSE applies to both regression architectures and is not an LSTM gate. Patient-macro MAE averages records within each patient, then weights patients equally.

Pending: no completed comparable research measurements.

Software pilot diagnostics (excluded from research rankings):

| name | split | mae_mV | rmse_mV | patient_macro_mae_mV | mse_mV2 | r2 | pearson_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.0005_h64_e2 | val | 0.116916 | 0.243421 | 0.115582 | 0.0592536 | -0.0022059 | 0.00407575 |
| cnn_lstm_P0_F1_lr0.0005_h64_e2 | val | 0.116873 | 0.243412 | 0.115542 | 0.0592493 | -0.00213214 | 0.002852 |
| cnn_lstm_P1_F0_lr0.0005_h64_e2 | val | 0.116915 | 0.24342 | 0.115581 | 0.0592535 | -0.00220412 | 0.00396794 |
| lstm_P0_F0_lr0.0005_h64_e2 | val | 0.116959 | 0.243398 | 0.11563 | 0.0592427 | -0.00202095 | 0.00629839 |
| lstm_P0_F1_lr0.0005_h64_e2 | val | 0.116903 | 0.243484 | 0.115575 | 0.0592844 | -0.00272596 | -0.00202754 |
| lstm_P1_F0_lr0.0005_h64_e2 | val | 0.116962 | 0.243403 | 0.115634 | 0.059245 | -0.00206024 | 0.00476671 |
| persistence | val | 0.147681 | 0.284758 | 0.147451 | 0.081087 | -0.371492 | 0.0461842 |
| repeat_window | val | 0.175421 | 0.344928 | 0.174014 | 0.118976 | -1.01233 | -0.0612501 |

## Twelve-lead prediction

Saved figures show each lead separately with observed history, original future truth, forecast, physical axes, and the forecast boundary. The pre-specified example is selected by metadata ID before errors; the median-error example is labelled. No fabricated peak markers or unvalidated P/Q/R/S/T delineation are used.

Pending: no completed comparable research measurements.

## Evidence-based comparison

Pending: no completed comparable research measurements.

Negative B−A favors B. Intervals resample patients and are descriptive after validation selection, using one seed without multiplicity correction. Medium/full cohorts and different horizons are not pooled. Low voltage error does not establish clinical validity; 100-Hz resolution and filter/morphology limitations remain. ARIMA is deferred. Feature-selection execution is pending unless feature_extension.json exists. Synthetic and pilot runs are software checks, not research performance evidence.
