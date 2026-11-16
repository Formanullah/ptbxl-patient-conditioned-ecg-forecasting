# PTB-XL research results

## Dataset

PTB-XL v1.0.3; source: https://physionet.org/content/ptb-xl/1.0.3/. Data root: `PTB-XL/records100`. WFDB reads physical mV, 100 Hz, twelve leads, 1,000 samples per record. History/target: `700/300`; batched input `[B,700,12]`, output `[B,300,12]`. This is within-record waveform forecasting, not clinical longitudinal prediction.

## Patient-safe folds

Official folds 1–8 train, 9 validate, 10 test; patients remain disjoint. Selected whole patients retain all eligible ECGs.

| split | records | patients | windows |
| --- | --- | --- | --- |
| train | 1731 | 1500 | 1731 |
| val | 2183 | 1942 | 2183 |

Reserved fold 10 metadata counts: {'records': 2198, 'patients': 1904}

Final-test ledger: pending / not enabled

## Preprocessing

P0 uses raw history; P1 uses causal 0.5–40 Hz SOS Butterworth filtering on observed history only (prototype N=4; bandpass order 8). Both use identical raw-training-history z-scores and predict the same original-mV future. Filtering may distort ST segments, QRS shape, phase, and initial samples; it is a hypothesis. History-only distortion/PSD figures do not establish physiological accuracy.

Measured validation filter effects:

| A | B | split | contrast | percentage_B_minus_A | difference_B_minus_A_mV | ci95_low | ci95_high | patients | bootstrap_samples | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.0005_h64_e60 | cnn_lstm_P1_F0_lr0.0005_h64_e60 | val | P0 versus P1 | 0.044678 | 5.07368e-05 | -0.00014501 | 0.000227234 | 1942 | 1000 | inconclusive |
| lstm_P0_F0_lr0.0005_h64_e60 | lstm_P1_F0_lr0.0005_h64_e60 | val | P0 versus P1 | 0.358096 | 0.000406187 | 0.000258577 | 0.000544774 | 1942 | 1000 | did not help |

## LSTM

Independent `LSTMForecaster`: optional early static embedding → stacked LSTM → final hidden state → MLP → all future samples. Exact active parameter counts and model settings are in each run.json; see docs/EQUATIONS.md.

| model | filter | features | static_values | hidden | layers | active_parameters | output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | P0 | permutation | 6 | 64 | 1 | 496896 | [B,300,12] |
| lstm | P0 | mrmr | 6 | 64 | 1 | 496896 | [B,300,12] |
| lstm | P0 | F0 | 0 | 64 | 1 | 492688 | [B,300,12] |
| lstm | P0 | expanded | 16 | 64 | 1 | 497056 | [B,300,12] |
| lstm | P1 | F0 | 0 | 64 | 1 | 492688 | [B,300,12] |
| lstm | P0 | F1 | 4 | 64 | 1 | 496864 | [B,300,12] |

## CNN-LSTM

Independent `CNNLSTMForecaster`: optional static channels → Conv1d(7,32) → ReLU → Conv1d(5,64) → ReLU → AvgPool(2) → LSTM → MLP. The primary 700-step history becomes 350 recurrent steps. No attention or contrastive projector is active.

| model | filter | features | static_values | hidden | layers | active_parameters | output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm | P0 | F0 | 0 | 64 | 1 | 519024 | [B,300,12] |
| cnn_lstm | P0 | permutation | 6 | 64 | 1 | 522720 | [B,300,12] |
| cnn_lstm | P1 | F0 | 0 | 64 | 1 | 519024 | [B,300,12] |
| cnn_lstm | P0 | F1 | 4 | 64 | 1 | 522688 | [B,300,12] |
| cnn_lstm | P0 | mrmr | 6 | 64 | 1 | 522720 | [B,300,12] |
| cnn_lstm | P0 | expanded | 16 | 64 | 1 | 522880 | [B,300,12] |

## Training

AdamW, normalized MSE, fixed learning rates, gradient clipping, seed 42; validation patient-macro MAE selects checkpoints. No early stopping truncates the budget.

| name | state | epochs_completed | best_epoch | runtime_seconds |
| --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.0001_h64_e60 | measured | 60 | 54 | 452.089 |
| cnn_lstm_P0_F0_lr0.0005_h64_e60 | measured | 60 | 16 | 557.055 |
| cnn_lstm_P0_F0_lr0.001_h64_e60 | measured | 60 | 8 | 449.844 |
| cnn_lstm_P0_F1_lr0.0005_h64_e60 | measured | 60 | 26 | 621.149 |
| cnn_lstm_P0_expanded_lr0.001_h64_e60 | measured | 60 | 8 | 649.191 |
| cnn_lstm_P0_mrmr_lr0.001_h64_e60 | measured | 60 | 5 | 649.216 |
| cnn_lstm_P0_permutation_lr0.001_h64_e60 | measured | 60 | 5 | 608.016 |
| cnn_lstm_P1_F0_lr0.0005_h64_e60 | measured | 60 | 9 | 534.424 |
| lstm_P0_F0_lr0.0001_h64_e60 | measured | 60 | 55 | 283.826 |
| lstm_P0_F0_lr0.0005_h64_e60 | measured | 60 | 12 | 276.083 |
| lstm_P0_F0_lr0.001_h64_e60 | measured | 60 | 5 | 264.492 |
| lstm_P0_F1_lr0.0005_h64_e60 | measured | 60 | 14 | 299.633 |
| lstm_P0_expanded_lr0.0005_h64_e60 | measured | 60 | 9 | 282.594 |
| lstm_P0_mrmr_lr0.0005_h64_e60 | measured | 60 | 9 | 278.932 |
| lstm_P0_permutation_lr0.0005_h64_e60 | measured | 60 | 9 | 278.865 |
| lstm_P1_F0_lr0.0005_h64_e60 | measured | 60 | 6 | 283.269 |
| persistence | measured | 0 | pending | pending |
| repeat_window | measured | 0 | pending | pending |

Recorded environments: `[{'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}, {'python': '3.13.5', 'torch': '2.15.0', 'device': 'cpu', 'gpu': None}]`. Full 120-epoch Colab runs remain pending unless their completed artifacts are present.

60/120 checkpoint evidence:

Pending: no completed comparable research measurements.

## RMSE/MAE

MAE = mean(|prediction − truth|), in mV; RMSE = sqrt(mean((prediction − truth)²)), in mV; MSE is mV². RMSE applies to both regression architectures and is not an LSTM gate. Patient-macro MAE averages records within each patient, then weights patients equally.

| name | split | mae_mV | rmse_mV | patient_macro_mae_mV | mse_mV2 | r2 | pearson_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.0001_h64_e60 | val | 0.114167 | 0.236132 | 0.113611 | 0.0557581 | 0.00318548 | 0.0662902 |
| cnn_lstm_P0_F0_lr0.0005_h64_e60 | val | 0.11414 | 0.236119 | 0.113561 | 0.0557519 | 0.00329557 | 0.0736856 |
| cnn_lstm_P0_F0_lr0.001_h64_e60 | val | 0.114026 | 0.236076 | 0.113446 | 0.0557321 | 0.00365096 | 0.0738621 |
| cnn_lstm_P0_F1_lr0.0005_h64_e60 | val | 0.114275 | 0.236374 | 0.113714 | 0.0558725 | 0.00114082 | 0.0759836 |
| cnn_lstm_P0_expanded_lr0.001_h64_e60 | val | 0.114056 | 0.235976 | 0.113483 | 0.0556846 | 0.00449948 | 0.0733333 |
| cnn_lstm_P0_mrmr_lr0.001_h64_e60 | val | 0.114029 | 0.236102 | 0.113485 | 0.0557439 | 0.0034388 | 0.0609083 |
| cnn_lstm_P0_permutation_lr0.001_h64_e60 | val | 0.114061 | 0.236118 | 0.113519 | 0.0557517 | 0.0032996 | 0.0567676 |
| cnn_lstm_P1_F0_lr0.0005_h64_e60 | val | 0.114149 | 0.236074 | 0.113612 | 0.0557309 | 0.00367213 | 0.0468705 |
| lstm_P0_F0_lr0.0001_h64_e60 | val | 0.114031 | 0.235946 | 0.113468 | 0.0556703 | 0.00475482 | 0.0759191 |
| lstm_P0_F0_lr0.0005_h64_e60 | val | 0.114002 | 0.235878 | 0.113429 | 0.0556384 | 0.00532523 | 0.0806669 |
| lstm_P0_F0_lr0.001_h64_e60 | val | 0.114003 | 0.23591 | 0.113447 | 0.0556533 | 0.00505874 | 0.0762107 |
| lstm_P0_F1_lr0.0005_h64_e60 | val | 0.114014 | 0.235896 | 0.113435 | 0.0556467 | 0.0051763 | 0.0825572 |
| lstm_P0_expanded_lr0.0005_h64_e60 | val | 0.114067 | 0.236013 | 0.113517 | 0.0557022 | 0.00418536 | 0.0692256 |
| lstm_P0_mrmr_lr0.0005_h64_e60 | val | 0.114055 | 0.236057 | 0.113497 | 0.0557227 | 0.00381802 | 0.0679639 |
| lstm_P0_permutation_lr0.0005_h64_e60 | val | 0.114001 | 0.236022 | 0.113449 | 0.0557066 | 0.00410655 | 0.0702252 |
| lstm_P1_F0_lr0.0005_h64_e60 | val | 0.114367 | 0.236264 | 0.113836 | 0.0558209 | 0.00206371 | 0.0343163 |
| persistence | val | 0.160413 | 0.314331 | 0.160367 | 0.0988042 | -0.76637 | 0.0299717 |
| repeat_window | val | 0.170941 | 0.326306 | 0.170786 | 0.106476 | -0.903519 | -0.0160786 |

## Twelve-lead prediction

Saved figures show each lead separately with observed history, original future truth, forecast, physical axes, and the forecast boundary. The pre-specified example is selected by metadata ID before errors; the median-error example is labelled. No fabricated peak markers or unvalidated P/Q/R/S/T delineation are used.

| lead | mae_mV | rmse_mV | pearson | name | split |
| --- | --- | --- | --- | --- | --- |
| I | 0.08486 | 0.152835 | 0.04429 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| II | 0.0965986 | 0.167539 | 0.0752482 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| III | 0.0870131 | 0.166458 | 0.0724611 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| aVR | 0.0800372 | 0.137021 | 0.0591839 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| aVL | 0.0718439 | 0.136102 | 0.0530171 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| aVF | 0.0802974 | 0.148346 | 0.0812636 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| V1 | 0.108083 | 0.221314 | 0.049336 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| V2 | 0.169581 | 0.342258 | 0.0829863 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| V3 | 0.173069 | 0.327743 | 0.0796502 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| V4 | 0.159898 | 0.3593 | 0.0289687 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| V5 | 0.137573 | 0.261046 | 0.0294568 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| V6 | 0.121149 | 0.244701 | 0.13962 | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val |
| I | 0.0852305 | 0.152993 | 0.0492235 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| II | 0.0962756 | 0.167159 | 0.0972676 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| III | 0.0870323 | 0.166251 | 0.0884445 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| aVR | 0.0800357 | 0.136982 | 0.0696148 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| aVL | 0.0721438 | 0.136158 | 0.0616375 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| aVF | 0.0800587 | 0.14803 | 0.101306 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| V1 | 0.108694 | 0.221227 | 0.0571941 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| V2 | 0.169619 | 0.341957 | 0.092641 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| V3 | 0.172971 | 0.327402 | 0.0913471 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| V4 | 0.159608 | 0.359649 | 0.023255 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| V5 | 0.137178 | 0.261148 | 0.0324924 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |
| V6 | 0.120839 | 0.245368 | 0.119803 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val |

## Evidence-based comparison

| A | B | split | contrast | percentage_B_minus_A | difference_B_minus_A_mV | ci95_low | ci95_high | patients | bootstrap_samples | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_P0_F0_lr0.0001_h64_e60 | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val | learning-rate comparison | -0.0444013 | -5.0445e-05 | -0.000136645 | 5.46881e-05 | 1942 | 1000 | inconclusive |
| cnn_lstm_P0_F0_lr0.0001_h64_e60 | cnn_lstm_P0_F0_lr0.001_h64_e60 | val | learning-rate comparison | -0.145304 | -0.000165082 | -0.000250456 | -8.16534e-05 | 1942 | 1000 | helped |
| cnn_lstm_P0_F0_lr0.0001_h64_e60 | lstm_P0_F0_lr0.0001_h64_e60 | val | matched architecture comparison | -0.126618 | -0.000143852 | -0.000269193 | -3.04959e-05 | 1942 | 1000 | helped |
| persistence | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val | neural pipeline versus raw-history baseline | -29.1552 | -0.0467553 | -0.0516834 | -0.0414592 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P0_F0_lr0.0001_h64_e60 | val | neural pipeline versus raw-history baseline | -33.4773 | -0.0571746 | -0.059106 | -0.0552202 | 1942 | 1000 | helped |
| cnn_lstm_P0_F0_lr0.0005_h64_e60 | cnn_lstm_P0_F0_lr0.001_h64_e60 | val | learning-rate comparison | -0.100947 | -0.000114637 | -0.000217962 | -2.27497e-05 | 1942 | 1000 | helped |
| cnn_lstm_P0_F0_lr0.0005_h64_e60 | cnn_lstm_P0_F1_lr0.0005_h64_e60 | val | static-feature ablation | 0.135091 | 0.00015341 | -1.47709e-05 | 0.000384975 | 1942 | 1000 | inconclusive |
| cnn_lstm_P0_F0_lr0.0005_h64_e60 | cnn_lstm_P1_F0_lr0.0005_h64_e60 | val | P0 versus P1 | 0.044678 | 5.07368e-05 | -0.00014501 | 0.000227234 | 1942 | 1000 | inconclusive |
| cnn_lstm_P0_F0_lr0.0005_h64_e60 | lstm_P0_F0_lr0.0005_h64_e60 | val | matched architecture comparison | -0.1159 | -0.000131618 | -0.000278544 | -6.40248e-06 | 1942 | 1000 | helped |
| persistence | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -29.1867 | -0.0468058 | -0.0517578 | -0.0414872 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P0_F0_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5068 | -0.057225 | -0.0591409 | -0.0552447 | 1942 | 1000 | helped |
| cnn_lstm_P0_F0_lr0.001_h64_e60 | cnn_lstm_P0_expanded_lr0.001_h64_e60 | val | static-feature ablation | 0.032424 | 3.67838e-05 | -6.18776e-05 | 0.000126722 | 1942 | 1000 | inconclusive |
| cnn_lstm_P0_F0_lr0.001_h64_e60 | cnn_lstm_P0_mrmr_lr0.001_h64_e60 | val | static-feature ablation | 0.0341674 | 3.87616e-05 | -8.75224e-05 | 0.000156697 | 1942 | 1000 | inconclusive |
| cnn_lstm_P0_F0_lr0.001_h64_e60 | cnn_lstm_P0_permutation_lr0.001_h64_e60 | val | static-feature ablation | 0.0641493 | 7.27751e-05 | -5.19701e-05 | 0.000186934 | 1942 | 1000 | inconclusive |
| cnn_lstm_P0_F0_lr0.001_h64_e60 | lstm_P0_F0_lr0.001_h64_e60 | val | matched architecture comparison | 0.000517147 | 5.86685e-07 | -0.000123609 | 0.000101632 | 1942 | 1000 | inconclusive |
| persistence | cnn_lstm_P0_F0_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2582 | -0.0469204 | -0.051954 | -0.0416146 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P0_F0_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -33.574 | -0.0573397 | -0.0592659 | -0.055354 | 1942 | 1000 | helped |
| cnn_lstm_P0_F1_lr0.0005_h64_e60 | lstm_P0_F1_lr0.0005_h64_e60 | val | matched architecture comparison | -0.246093 | -0.000279843 | -0.000602592 | -3.66887e-05 | 1942 | 1000 | helped |
| persistence | cnn_lstm_P0_F1_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -29.091 | -0.0466524 | -0.0515358 | -0.0413558 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P0_F1_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -33.417 | -0.0570716 | -0.0590507 | -0.0549898 | 1942 | 1000 | helped |
| cnn_lstm_P0_expanded_lr0.001_h64_e60 | cnn_lstm_P0_mrmr_lr0.001_h64_e60 | val | static-feature ablation | 0.00174282 | 1.97781e-06 | -9.8403e-05 | 0.000103549 | 1942 | 1000 | inconclusive |
| cnn_lstm_P0_expanded_lr0.001_h64_e60 | cnn_lstm_P0_permutation_lr0.001_h64_e60 | val | static-feature ablation | 0.031715 | 3.59912e-05 | -5.78524e-05 | 0.000137409 | 1942 | 1000 | inconclusive |
| persistence | cnn_lstm_P0_expanded_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2352 | -0.0468836 | -0.0518729 | -0.0415766 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P0_expanded_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5524 | -0.0573029 | -0.059205 | -0.0553261 | 1942 | 1000 | helped |
| cnn_lstm_P0_mrmr_lr0.001_h64_e60 | cnn_lstm_P0_permutation_lr0.001_h64_e60 | val | static-feature ablation | 0.0299717 | 3.40134e-05 | 1.42319e-05 | 5.35138e-05 | 1942 | 1000 | did not help |
| persistence | cnn_lstm_P0_mrmr_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -29.234 | -0.0468817 | -0.051844 | -0.041645 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P0_mrmr_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5513 | -0.0573009 | -0.0592036 | -0.0553462 | 1942 | 1000 | helped |
| persistence | cnn_lstm_P0_permutation_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2128 | -0.0468476 | -0.0518074 | -0.0416131 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P0_permutation_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5314 | -0.0572669 | -0.0591802 | -0.0553036 | 1942 | 1000 | helped |
| cnn_lstm_P1_F0_lr0.0005_h64_e60 | lstm_P1_F0_lr0.0005_h64_e60 | val | matched architecture comparison | 0.197015 | 0.000223832 | 0.000166692 | 0.000277235 | 1942 | 1000 | did not help |
| persistence | cnn_lstm_P1_F0_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -29.1551 | -0.046755 | -0.0517388 | -0.0414751 | 1942 | 1000 | helped |
| repeat_window | cnn_lstm_P1_F0_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -33.4771 | -0.0571743 | -0.0590723 | -0.055219 | 1942 | 1000 | helped |
| lstm_P0_F0_lr0.0001_h64_e60 | lstm_P0_F0_lr0.0005_h64_e60 | val | learning-rate comparison | -0.0336752 | -3.82105e-05 | -7.77837e-05 | 6.38234e-06 | 1942 | 1000 | inconclusive |
| lstm_P0_F0_lr0.0001_h64_e60 | lstm_P0_F0_lr0.001_h64_e60 | val | learning-rate comparison | -0.0181927 | -2.06429e-05 | -7.47428e-05 | 2.10843e-05 | 1942 | 1000 | inconclusive |
| persistence | lstm_P0_F0_lr0.0001_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2449 | -0.0468992 | -0.0518566 | -0.0416212 | 1942 | 1000 | helped |
| repeat_window | lstm_P0_F0_lr0.0001_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5615 | -0.0573184 | -0.0592518 | -0.0553504 | 1942 | 1000 | helped |
| lstm_P0_F0_lr0.0005_h64_e60 | lstm_P0_F0_lr0.001_h64_e60 | val | learning-rate comparison | 0.0154877 | 1.75676e-05 | -5.5267e-05 | 7.1745e-05 | 1942 | 1000 | inconclusive |
| lstm_P0_F0_lr0.0005_h64_e60 | lstm_P0_F1_lr0.0005_h64_e60 | val | static-feature ablation | 0.00457105 | 5.18492e-06 | -7.14909e-05 | 8.16098e-05 | 1942 | 1000 | inconclusive |
| lstm_P0_F0_lr0.0005_h64_e60 | lstm_P0_expanded_lr0.0005_h64_e60 | val | static-feature ablation | 0.0770087 | 8.73505e-05 | -1.10699e-05 | 0.000177054 | 1942 | 1000 | inconclusive |
| lstm_P0_F0_lr0.0005_h64_e60 | lstm_P0_mrmr_lr0.0005_h64_e60 | val | static-feature ablation | 0.0594965 | 6.74866e-05 | -2.03193e-05 | 0.000148248 | 1942 | 1000 | inconclusive |
| lstm_P0_F0_lr0.0005_h64_e60 | lstm_P0_permutation_lr0.0005_h64_e60 | val | static-feature ablation | 0.0169025 | 1.91724e-05 | -6.57664e-05 | 9.77242e-05 | 1942 | 1000 | inconclusive |
| lstm_P0_F0_lr0.0005_h64_e60 | lstm_P1_F0_lr0.0005_h64_e60 | val | P0 versus P1 | 0.358096 | 0.000406187 | 0.000258577 | 0.000544774 | 1942 | 1000 | did not help |
| persistence | lstm_P0_F0_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2688 | -0.0469374 | -0.0519038 | -0.0416454 | 1942 | 1000 | helped |
| repeat_window | lstm_P0_F0_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5839 | -0.0573566 | -0.0592931 | -0.0553881 | 1942 | 1000 | helped |
| persistence | lstm_P0_F0_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2578 | -0.0469198 | -0.0518951 | -0.041643 | 1942 | 1000 | helped |
| repeat_window | lstm_P0_F0_lr0.001_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5736 | -0.0573391 | -0.0592648 | -0.0553768 | 1942 | 1000 | helped |
| lstm_P0_F1_lr0.0005_h64_e60 | lstm_P0_expanded_lr0.0005_h64_e60 | val | static-feature ablation | 0.0724343 | 8.21656e-05 | -1.16173e-05 | 0.000170547 | 1942 | 1000 | inconclusive |
| lstm_P0_F1_lr0.0005_h64_e60 | lstm_P0_mrmr_lr0.0005_h64_e60 | val | static-feature ablation | 0.0549229 | 6.23016e-05 | -2.91447e-05 | 0.000140482 | 1942 | 1000 | inconclusive |
| lstm_P0_F1_lr0.0005_h64_e60 | lstm_P0_permutation_lr0.0005_h64_e60 | val | static-feature ablation | 0.0123308 | 1.39874e-05 | -7.73865e-05 | 9.30643e-05 | 1942 | 1000 | inconclusive |
| persistence | lstm_P0_F1_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2655 | -0.0469322 | -0.0519367 | -0.0416551 | 1942 | 1000 | helped |
| repeat_window | lstm_P0_F1_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5809 | -0.0573515 | -0.0592499 | -0.055409 | 1942 | 1000 | helped |
| lstm_P0_expanded_lr0.0005_h64_e60 | lstm_P0_mrmr_lr0.0005_h64_e60 | val | static-feature ablation | -0.0174987 | -1.9864e-05 | -5.34506e-05 | 1.69033e-05 | 1942 | 1000 | inconclusive |
| lstm_P0_expanded_lr0.0005_h64_e60 | lstm_P0_permutation_lr0.0005_h64_e60 | val | static-feature ablation | -0.06006 | -6.81782e-05 | -0.00010303 | -3.204e-05 | 1942 | 1000 | helped |
| persistence | lstm_P0_expanded_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2143 | -0.04685 | -0.051818 | -0.0416064 | 1942 | 1000 | helped |
| repeat_window | lstm_P0_expanded_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5328 | -0.0572693 | -0.0591756 | -0.0553378 | 1942 | 1000 | helped |
| lstm_P0_mrmr_lr0.0005_h64_e60 | lstm_P0_permutation_lr0.0005_h64_e60 | val | static-feature ablation | -0.0425687 | -4.83142e-05 | -6.77309e-05 | -3.07302e-05 | 1942 | 1000 | helped |
| persistence | lstm_P0_mrmr_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2267 | -0.0468699 | -0.0518344 | -0.0416044 | 1942 | 1000 | helped |
| repeat_window | lstm_P0_mrmr_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5444 | -0.0572892 | -0.059201 | -0.0553592 | 1942 | 1000 | helped |
| persistence | lstm_P0_permutation_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -29.2568 | -0.0469182 | -0.0518862 | -0.0416508 | 1942 | 1000 | helped |
| repeat_window | lstm_P0_permutation_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -33.5727 | -0.0573375 | -0.0592565 | -0.055412 | 1942 | 1000 | helped |
| persistence | lstm_P1_F0_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -29.0155 | -0.0465312 | -0.0514903 | -0.0412381 | 1942 | 1000 | helped |
| repeat_window | lstm_P1_F0_lr0.0005_h64_e60 | val | neural pipeline versus raw-history baseline | -33.3461 | -0.0569505 | -0.0588544 | -0.0550117 | 1942 | 1000 | helped |

Negative B−A favors B. Intervals resample patients and are descriptive after validation selection, using one seed without multiplicity correction. Medium/full cohorts and different horizons are not pooled. Low voltage error does not establish clinical validity; 100-Hz resolution and filter/morphology limitations remain. ARIMA is deferred. Feature-selection execution is pending unless feature_extension.json exists. Synthetic and pilot runs are software checks, not research performance evidence.
