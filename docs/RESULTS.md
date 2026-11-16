# PTB-XL research results

## Dataset

PTB-XL v1.0.3; source: https://physionet.org/content/ptb-xl/1.0.3/. Data root: `PTB-XL/records100`. WFDB reads physical mV, 100 Hz, twelve leads, 1,000 samples per record. History/target: `700/300`; batched input `[B,700,12]`, output `[B,300,12]`. This is within-record waveform forecasting, not clinical longitudinal prediction.

## Patient-safe folds

Official folds 1–8 train, 9 validate, 10 test; patients remain disjoint. Selected whole patients retain all eligible ECGs.

| split | records | patients | windows |
| --- | --- | --- | --- |
| train | 17418 | 15023 | 17418 |
| val | 2183 | 1942 | 2183 |

Reserved fold 10 metadata counts: {'records': 2198, 'patients': 1904}

Final-test ledger: pending / not enabled

## Preprocessing

P0 uses raw history; P1 uses causal 0.5–40 Hz SOS Butterworth filtering on observed history only (prototype N=4; bandpass order 8). Both use identical raw-training-history z-scores and predict the same original-mV future. Filtering may distort ST segments, QRS shape, phase, and initial samples; it is a hypothesis. History-only distortion/PSD figures do not establish physiological accuracy.

Measured validation filter effects:

| A | B | split | contrast | percentage_B_minus_A | difference_B_minus_A_mV | ci95_low | ci95_high | patients | bootstrap_samples | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.001_h64_e120 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val | P0 versus P1 | 3.35539 | 0.00352971 | 0.00270689 | 0.00420521 | 1942 | 1000 | did not help |
| lstm_P0_F0_lr0.0005_h64_e120 | lstm_P1_F0_lr0.0005_h64_e120 | val | P0 versus P1 | -0.822759 | -0.000914081 | -0.00141262 | -0.000440768 | 1942 | 1000 | helped |

## LSTM

Independent `LSTMForecaster`: optional early static embedding → stacked LSTM → final hidden state → MLP → all future samples. Exact active parameter counts and model settings are in each run.json; see docs/EQUATIONS.md.

| model | filter | features | static_values | hidden | layers | active_parameters | output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | P1 | F0 | 0 | 64 | 1 | 492688 | [B,300,12] |
| lstm | P0 | F0 | 0 | 64 | 1 | 492688 | [B,300,12] |

## CNN-LSTM

Independent `CNNLSTMForecaster`: optional static channels → Conv1d(7,32) → ReLU → Conv1d(5,64) → ReLU → AvgPool(2) → LSTM → MLP. The primary 700-step history becomes 350 recurrent steps. No attention or contrastive projector is active.

| model | filter | features | static_values | hidden | layers | active_parameters | output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm | P0 | F0 | 0 | 64 | 1 | 519024 | [B,300,12] |
| cnn_lstm | P1 | F0 | 0 | 64 | 1 | 519024 | [B,300,12] |

## Training

AdamW, normalized MSE, fixed learning rates, gradient clipping, seed 42; validation patient-macro MAE selects checkpoints. No early stopping truncates the budget.

| name | state | epochs_completed | best_epoch | runtime_seconds |
| --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.001_h64_e120 | measured | 120 | 116 | 5714.68 |
| cnn_lstm_P1_F0_lr0.001_h64_e120 | measured | 120 | 43 | 6136.42 |
| lstm_P0_F0_lr0.0005_h64_e120 | measured | 120 | 29 | 3678.08 |
| lstm_P1_F0_lr0.0005_h64_e120 | measured | 120 | 99 | 4265.77 |
| persistence | measured | 0 | pending | pending |
| repeat_window | measured | 0 | pending | pending |

Recorded environments: `[{'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}]`. Full 120-epoch Colab runs remain pending unless their completed artifacts are present.

60/120 checkpoint evidence:

| name | best_epoch | patient_macro_mae_mV | rmse_mV |
| --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.001_h64_e120_through_60 | 54 | 0.106835 | 0.228533 |
| cnn_lstm_P0_F0_lr0.001_h64_e120_through_120 | 116 | 0.105195 | 0.226551 |
| cnn_lstm_P1_F0_lr0.001_h64_e120_through_60 | 43 | 0.108725 | 0.227847 |
| lstm_P0_F0_lr0.0005_h64_e120_through_60 | 29 | 0.111099 | 0.234362 |
| lstm_P1_F0_lr0.0005_h64_e120_through_120 | 99 | 0.110185 | 0.231235 |
| lstm_P1_F0_lr0.0005_h64_e120_through_60 | 60 | 0.110721 | 0.231858 |
| lstm_P0_F0_lr0.0005_h64_e120_through_120 | 29 | 0.111099 | 0.234362 |
| cnn_lstm_P1_F0_lr0.001_h64_e120_through_120 | 43 | 0.108725 | 0.227847 |

## RMSE/MAE

MAE = mean(|prediction − truth|), in mV; RMSE = sqrt(mean((prediction − truth)²)), in mV; MSE is mV². RMSE applies to both regression architectures and is not an LSTM gate. Patient-macro MAE averages records within each patient, then weights patients equally.

| name | split | mae_mV | rmse_mV | patient_macro_mae_mV | mse_mV2 | r2 | pearson_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.001_h64_e120 | val | 0.105776 | 0.226551 | 0.105195 | 0.0513253 | 0.0824337 | 0.317735 |
| cnn_lstm_P1_F0_lr0.001_h64_e120 | val | 0.109367 | 0.227847 | 0.108725 | 0.0519141 | 0.0719068 | 0.263955 |
| lstm_P0_F0_lr0.0005_h64_e120 | val | 0.111684 | 0.234362 | 0.111099 | 0.0549258 | 0.0180654 | 0.147086 |
| lstm_P1_F0_lr0.0005_h64_e120 | val | 0.110832 | 0.231235 | 0.110185 | 0.0534696 | 0.0440975 | 0.20405 |
| persistence | val | 0.160413 | 0.314331 | 0.160367 | 0.0988042 | -0.76637 | 0.0299717 |
| repeat_window | val | 0.170941 | 0.326306 | 0.170786 | 0.106476 | -0.903519 | -0.0160786 |

## Twelve-lead prediction

Saved figures show each lead separately with observed history, original future truth, forecast, physical axes, and the forecast boundary. The pre-specified example is selected by metadata ID before errors; the median-error example is labelled. No fabricated peak markers or unvalidated P/Q/R/S/T delineation are used.

| lead | mae_mV | rmse_mV | pearson | name | split |
| --- | --- | --- | --- | --- | --- |
| I | 0.0770903 | 0.1429 | 0.353696 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| II | 0.0861936 | 0.157102 | 0.354698 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| III | 0.0760788 | 0.152906 | 0.39987 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| aVR | 0.0725575 | 0.129251 | 0.33519 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| aVL | 0.0638178 | 0.125421 | 0.388095 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| aVF | 0.0706 | 0.137457 | 0.384246 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| V1 | 0.105175 | 0.208046 | 0.34242 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| V2 | 0.158921 | 0.32513 | 0.320515 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| V3 | 0.16081 | 0.31451 | 0.290681 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| V4 | 0.149634 | 0.348041 | 0.248916 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| V5 | 0.131431 | 0.253395 | 0.233852 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| V6 | 0.117 | 0.250679 | 0.160637 | cnn_lstm_P0_F0_lr0.001_h64_e120 | val |
| I | 0.0815095 | 0.146175 | 0.291723 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| II | 0.0929232 | 0.163095 | 0.239072 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| III | 0.0866757 | 0.160039 | 0.281849 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| aVR | 0.0763825 | 0.132603 | 0.257225 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| aVL | 0.0709194 | 0.129785 | 0.302312 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| aVF | 0.0789358 | 0.14396 | 0.252535 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| V1 | 0.105779 | 0.211499 | 0.293887 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| V2 | 0.157719 | 0.32261 | 0.34174 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| V3 | 0.159695 | 0.313173 | 0.302244 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| V4 | 0.150113 | 0.351883 | 0.195545 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| V5 | 0.132316 | 0.254535 | 0.21471 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |
| V6 | 0.11944 | 0.242366 | 0.194625 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val |

## Evidence-based comparison

| A | B | split | contrast | percentage_B_minus_A | difference_B_minus_A_mV | ci95_low | ci95_high | patients | bootstrap_samples | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.001_h64_e120 | cnn_lstm_P1_F0_lr0.001_h64_e120 | val | P0 versus P1 | 3.35539 | 0.00352971 | 0.00270689 | 0.00420521 | 1942 | 1000 | did not help |
| cnn_lstm_P0_F0_lr0.001_h64_e120 | lstm_P0_F0_lr0.0005_h64_e120 | val | selected pipelines (architecture and recipe may differ) | 5.61272 | 0.0059043 | 0.00520167 | 0.00653392 | 1942 | 1000 | did not help |
| persistence | cnn_lstm_P0_F0_lr0.001_h64_e120 | val | neural pipeline versus raw-history baseline | -34.4035 | -0.0551717 | -0.0601215 | -0.0498423 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P0_F0_lr0.001_h64_e120 | val | neural pipeline versus raw-history baseline | -38.4053 | -0.065591 | -0.0677004 | -0.0632149 | 1942 | 1000 | helped |
| cnn_lstm_P1_F0_lr0.001_h64_e120 | lstm_P1_F0_lr0.0005_h64_e120 | val | selected pipelines (architecture and recipe may differ) | 1.34331 | 0.00146052 | 0.00114414 | 0.0018212 | 1942 | 1000 | did not help |
| persistence | cnn_lstm_P1_F0_lr0.001_h64_e120 | val | neural pipeline versus raw-history baseline | -32.2024 | -0.051642 | -0.0566419 | -0.0463788 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P1_F0_lr0.001_h64_e120 | val | neural pipeline versus raw-history baseline | -36.3386 | -0.0620613 | -0.0639733 | -0.0600187 | 1942 | 1000 | helped |
| lstm_P0_F0_lr0.0005_h64_e120 | lstm_P1_F0_lr0.0005_h64_e120 | val | P0 versus P1 | -0.822759 | -0.000914081 | -0.00141262 | -0.000440768 | 1942 | 1000 | helped |
| persistence | lstm_P0_F0_lr0.0005_h64_e120 | val | neural pipeline versus raw-history baseline | -30.7217 | -0.0492674 | -0.0542487 | -0.0439978 | 1942 | 1000 | helped |
| repeat_window | lstm_P0_F0_lr0.0005_h64_e120 | val | neural pipeline versus raw-history baseline | -34.9482 | -0.0596867 | -0.0616316 | -0.0575688 | 1942 | 1000 | helped |
| persistence | lstm_P1_F0_lr0.0005_h64_e120 | val | neural pipeline versus raw-history baseline | -31.2917 | -0.0501815 | -0.0550372 | -0.045017 | 1942 | 1000 | helped |
| repeat_window | lstm_P1_F0_lr0.0005_h64_e120 | val | neural pipeline versus raw-history baseline | -35.4834 | -0.0606008 | -0.06248 | -0.0586728 | 1942 | 1000 | helped |

Negative B−A favors B. Intervals resample patients and are descriptive after validation selection, using one seed without multiplicity correction. Medium/full cohorts and different horizons are not pooled. Low voltage error does not establish clinical validity; 100-Hz resolution and filter/morphology limitations remain. ARIMA is deferred. Feature-selection execution is pending unless feature_extension.json exists. Synthetic and pilot runs are software checks, not research performance evidence.
