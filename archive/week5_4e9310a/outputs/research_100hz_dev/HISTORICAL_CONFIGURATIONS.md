# Historical configurations — not active training presets

## base.yaml

```yaml
# Complete new 100 Hz cohort. Fold 10 stays blocked until final selection is frozen.
extends: research.yaml
profile: research_100hz_full
data:
  cache: data/research_100hz_full_p0
subset:
  mode: full
  seed: 42
paths:
  experiments: experiments/research_100hz_full
  outputs: outputs/research_100hz_full
  documents: Project_Documents/research_100hz_full

```

## final_full.yaml

```yaml
profile: research_100hz_full
data:
  root: PTB-XL
  cache: data/research_100hz_full_p3
  sampling_rate: 100
  history_seconds: 5.0
  horizon_seconds: 3.0
  window_stride_seconds: 1.0
  start: 0
  layout: time_first
  history: 500
  horizon: 300
  stride: 100
subset:
  mode: full
  seed: 42
  patients:
    train: 450
    val: 100
    test: 100
preprocessing:
  filter: none
  scaling: robust
  order: 4
  low_hz: 0.5
  high_hz: 40.0
features: {}
model:
  hidden: 64
  depth: 1
  dropout: 0.2
  static_embedding: 16
  fusion: late
  kernels:
  - 7
  - 5
  - 3
  pool: 2
  residual: false
  head_width: 256
training:
  seed: 42
  epochs: 60
  early_stopping: false
  patience: 10
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  gradient_clip: 1.0
  num_workers: 0
  threads: 4
  device: auto
  loss: mse
  derivative_weight: 0.0
  spectral_weight: 0.0
  scheduler: none
contrastive:
  weight: 0.1
  temperature: 0.1
  jitter_std: 0.01
  gain_min: 0.95
  gain_max: 1.05
evaluation:
  bootstrap_samples: 1000
  peak_tolerance_ms: 100
  allow_test: false
paths:
  experiments: experiments/research_100hz_full
  outputs: outputs/research_100hz_full
  documents: Project_Documents/research_100hz_full
experiments:
  full_lstm_60:
    experiment:
      architecture: lstm
      variant: ecg_only
      stage: full_finalist
    training:
      epochs: 60
      early_stopping: false
  full_lstm_100:
    experiment:
      architecture: lstm
      variant: ecg_only
      stage: full_finalist
    training:
      epochs: 100
      early_stopping: false
  full_original_lstm_60:
    experiment:
      architecture: lstm
      variant: ecg_only
      stage: full_reference
    model:
      hidden: 128
      depth: 2
      static_embedding: 32
    training:
      epochs: 60
  full_lstm_history_60:
    experiment:
      architecture: lstm
      variant: patient_history
      stage: full_conditioning
    training:
      epochs: 60
  full_lstm_early_60:
    experiment:
      architecture: lstm
      variant: patient_history
      stage: full_fusion
    model:
      fusion: early
    training:
      epochs: 60
  full_cnn_60:
    experiment:
      architecture: cnn_lstm
      variant: ecg_only
      stage: full_finalist
    training:
      epochs: 60
      early_stopping: false
  full_cnn_100:
    experiment:
      architecture: cnn_lstm
      variant: ecg_only
      stage: full_finalist
    training:
      epochs: 100
      early_stopping: false
  full_original_cnn_60:
    experiment:
      architecture: cnn_lstm
      variant: ecg_only
      stage: full_reference
    model:
      hidden: 128
      depth: 2
      static_embedding: 32
    training:
      epochs: 60
  full_cnn_history_60:
    experiment:
      architecture: cnn_lstm
      variant: patient_history
      stage: full_conditioning
    training:
      epochs: 60
  full_cnn_early_60:
    experiment:
      architecture: cnn_lstm
      variant: patient_history
      stage: full_fusion
    model:
      fusion: early
    training:
      epochs: 60
  full_lstm_expanded_60:
    experiment:
      architecture: lstm
      variant: expanded
      stage: full_features
    training:
      epochs: 60
  full_lstm_mrmr_60:
    experiment:
      architecture: lstm
      variant: mrmr
      stage: full_features
    training:
      epochs: 60
    features:
      mrmr_path: outputs/research_100hz_full/mrmr.json
  full_lstm_permutation_60:
    experiment:
      architecture: lstm
      variant: permutation
      stage: full_features
    training:
      epochs: 60
    features:
      permutation_path: outputs/research_100hz_full/permutation.json
  full_lstm_p0_60:
    experiment:
      architecture: lstm
      variant: ecg_only
      stage: full_preprocessing
    training:
      epochs: 60
    data:
      cache: data/research_100hz_full_p0
    preprocessing:
      scaling: zscore

```

## finalists.yaml

```yaml
profile: research_100hz_dev
data:
  root: PTB-XL
  cache: data/research_100hz_dev_p3
  sampling_rate: 100
  history_seconds: 5.0
  horizon_seconds: 3.0
  window_stride_seconds: 1.0
  start: 0
  layout: time_first
  history: 500
  horizon: 300
  stride: 100
subset:
  mode: custom
  seed: 42
  patients:
    train: 64
    val: 24
    test: 16
preprocessing:
  filter: none
  scaling: robust
  order: 4
  low_hz: 0.5
  high_hz: 40.0
features: {}
model:
  hidden: 64
  depth: 1
  dropout: 0.2
  static_embedding: 16
  fusion: late
  kernels:
  - 7
  - 5
  - 3
  pool: 2
  residual: false
  head_width: 256
training:
  seed: 42
  epochs: 8
  early_stopping: false
  patience: 10
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  gradient_clip: 1.0
  num_workers: 0
  threads: 4
  device: auto
  loss: mse
  derivative_weight: 0.0
  spectral_weight: 0.0
  scheduler: none
contrastive:
  weight: 0.1
  temperature: 0.1
  jitter_std: 0.01
  gain_min: 0.95
  gain_max: 1.05
evaluation:
  bootstrap_samples: 1000
  peak_tolerance_ms: 100
  allow_test: false
paths:
  experiments: experiments/research_100hz_dev
  outputs: outputs/research_100hz_dev
  documents: Project_Documents/research_100hz
experiments:
  lstm_60:
    experiment:
      architecture: lstm
      variant: ecg_only
      stage: budget
      screening_source: p3_lstm
    model: &id001
      hidden: 64
      depth: 1
      dropout: 0.2
      static_embedding: 16
      fusion: late
      kernels:
      - 7
      - 5
      - 3
      pool: 2
      residual: false
      head_width: 256
    training:
      seed: 42
      epochs: 60
      early_stopping: false
      patience: 10
      batch_size: 64
      lr: 0.001
      weight_decay: 0.0001
      gradient_clip: 1.0
      num_workers: 0
      threads: 4
      device: auto
      loss: mse
      derivative_weight: 0.0
      spectral_weight: 0.0
      scheduler: none
    features: &id002 {}
  lstm_100:
    experiment:
      architecture: lstm
      variant: ecg_only
      stage: budget
      screening_source: p3_lstm
    model: *id001
    training:
      seed: 42
      epochs: 100
      early_stopping: false
      patience: 10
      batch_size: 64
      lr: 0.001
      weight_decay: 0.0001
      gradient_clip: 1.0
      num_workers: 0
      threads: 4
      device: auto
      loss: mse
      derivative_weight: 0.0
      spectral_weight: 0.0
      scheduler: none
    features: *id002
  cnn_60:
    experiment:
      architecture: cnn_lstm
      variant: ecg_only
      stage: budget
      screening_source: cnn_ecg
    model: &id003
      hidden: 64
      depth: 1
      dropout: 0.2
      static_embedding: 16
      fusion: late
      kernels:
      - 7
      - 5
      - 3
      pool: 2
      residual: false
      head_width: 256
    training:
      seed: 42
      epochs: 60
      early_stopping: false
      patience: 10
      batch_size: 64
      lr: 0.001
      weight_decay: 0.0001
      gradient_clip: 1.0
      num_workers: 0
      threads: 4
      device: auto
      loss: mse
      derivative_weight: 0.0
      spectral_weight: 0.0
      scheduler: none
    features: &id004 {}
  cnn_100:
    experiment:
      architecture: cnn_lstm
      variant: ecg_only
      stage: budget
      screening_source: cnn_ecg
    model: *id003
    training:
      seed: 42
      epochs: 100
      early_stopping: false
      patience: 10
      batch_size: 64
      lr: 0.001
      weight_decay: 0.0001
      gradient_clip: 1.0
      num_workers: 0
      threads: 4
      device: auto
      loss: mse
      derivative_weight: 0.0
      spectral_weight: 0.0
      scheduler: none
    features: *id004

```

## legacy_500hz.yaml

```yaml
data:
  root: PTB-XL
  cache: data/preprocessed
  sampling_rate: 500
  history: 300
  horizon: 150
  start: 0
model:
  hidden: 128
  depth: 2
  dropout: 0.2
  static_embedding: 32
training:
  seed: 42
  epochs: 25
  patience: 5
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  gradient_clip: 1.0
  num_workers: 0
  threads: 4
  device: auto
contrastive:
  weight: 0.1
  temperature: 0.1
  jitter_std: 0.01
  gain_min: 0.95
  gain_max: 1.05
evaluation:
  bootstrap_samples: 1000
paths:
  experiments: experiments
  outputs: outputs
  documents: Project_Documents/full_data

```

## local.yaml

```yaml
# Editable primary local experiment: 100 Hz, 5 s input, 3 s forecast.
# study.yaml selects the smaller measured CPU development cohort.
extends: research.yaml
profile: research_100hz_local
paths:
  experiments: experiments/research_100hz_local
  outputs: outputs/research_100hz_local
  documents: Project_Documents/research_100hz_local

```

## medium.yaml

```yaml
extends: research.yaml
profile: research_100hz_medium
data:
  cache: data/research_100hz_medium_p0
subset:
  mode: medium
  seed: 42
  patients: {train: 1500, val: 300, test: 100}
paths:
  experiments: experiments/research_100hz_medium
  outputs: outputs/research_100hz_medium
  documents: Project_Documents/research_100hz_medium

```

## pilot.yaml

```yaml
profile: pilot
data:
  root: PTB-XL
  cache: data/pilot
  sampling_rate: 500
  history: 300
  horizon: 150
  start: 0
subset:
  source_cache: data/preprocessed
  seed: 42
  patients:
    train: 450
    val: 100
    test: 100
model:
  hidden: 128
  depth: 2
  dropout: 0.2
  static_embedding: 32
training:
  seed: 42
  epochs: 5
  patience: 3
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  gradient_clip: 1.0
  num_workers: 0
  threads: 4
  device: auto
contrastive:
  weight: 0.1
  temperature: 0.1
  jitter_std: 0.01
  gain_min: 0.95
  gain_max: 1.05
evaluation:
  bootstrap_samples: 1000
paths:
  experiments: experiments/pilot
  outputs: outputs/pilot
  documents: Project_Documents

```

## research.yaml

```yaml
# Primary new protocol. Explicit manual commands; no test evaluation during development.
profile: research_100hz_pilot
data:
  root: PTB-XL
  cache: data/research_100hz_pilot_p0
  sampling_rate: 100
  history_seconds: 5.0
  horizon_seconds: 3.0
  window_stride_seconds: 1.0
  start: 0
  layout: time_first
subset:
  mode: pilot
  seed: 42
  patients: {train: 450, val: 100, test: 100}
preprocessing:
  filter: none
  scaling: zscore
  order: 4
  low_hz: 0.5
  high_hz: 40.0
features: {}
model:
  hidden: 128
  depth: 2
  dropout: 0.2
  static_embedding: 32
  fusion: late
  kernels: [7, 5, 3]
  pool: 2
  residual: false
  head_width: 256
training:
  seed: 42
  epochs: 60
  early_stopping: false
  patience: 10
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  gradient_clip: 1.0
  num_workers: 0
  threads: 4
  device: auto
  loss: mse
  derivative_weight: 0.0
  spectral_weight: 0.0
  scheduler: none
contrastive:
  weight: 0.1
  temperature: 0.1
  jitter_std: 0.01
  gain_min: 0.95
  gain_max: 1.05
evaluation:
  bootstrap_samples: 1000
  peak_tolerance_ms: 100
  allow_test: false
paths:
  experiments: experiments/research_100hz_pilot
  outputs: outputs/research_100hz_pilot
  documents: Project_Documents/research_100hz

```

## selected_study.yaml

```yaml
# Use only for the stages after the P0-P3 preprocessing comparison.
# Validation selected no filter + robust scaling; P4 is identical to measured P3.
extends: study.yaml
data:
  cache: data/research_100hz_dev_p3
preprocessing:
  filter: none
  scaling: robust

```

## study.yaml

```yaml
# CPU development subset. These are explicit recipes, not an automatic run-all command.
# Run e.g. python src/train.py --config configs/study.yaml --experiment p0_lstm
extends: research.yaml
profile: research_100hz_dev
data:
  cache: data/research_100hz_dev_p0
subset:
  mode: custom
  seed: 42
  patients: {train: 64, val: 24, test: 16}
model:
  hidden: 64
  depth: 1
  static_embedding: 16
training:
  epochs: 8
paths:
  experiments: experiments/research_100hz_dev
  outputs: outputs/research_100hz_dev
  documents: Project_Documents/research_100hz
experiment: {architecture: lstm, variant: ecg_only, stage: preprocessing}
experiments:
  p0_lstm: {}
  p1_lstm:
    data: {cache: data/research_100hz_dev_p1}
    preprocessing: {filter: highpass}
  p2_lstm:
    data: {cache: data/research_100hz_dev_p2}
    preprocessing: {filter: bandpass}
  p3_lstm:
    data: {cache: data/research_100hz_dev_p3}
    preprocessing: {scaling: robust}
  original_lstm:
    experiment: {stage: reference}
    model: {hidden: 128, depth: 2, static_embedding: 32}
  original_cnn_lstm:
    experiment: {architecture: cnn_lstm, stage: reference}
    model: {hidden: 128, depth: 2, static_embedding: 32}
  cnn_ecg:
    experiment: {architecture: cnn_lstm, stage: architecture}
  lstm_demographics:
    experiment: {variant: demographics, stage: features}
  lstm_history:
    experiment: {variant: patient_history, stage: features}
  lstm_expanded:
    experiment: {variant: expanded, stage: features}
  lstm_mrmr:
    experiment: {variant: mrmr, stage: features}
    features: {mrmr_path: outputs/research_100hz_dev/mrmr.json}
  lstm_permutation:
    experiment: {variant: permutation, stage: features}
    features: {permutation_path: outputs/research_100hz_dev/permutation.json}
  lstm_early:
    experiment: {variant: patient_history, stage: fusion}
    model: {fusion: early}
  cnn_history:
    experiment: {architecture: cnn_lstm, variant: patient_history, stage: fusion}
  cnn_early:
    experiment: {architecture: cnn_lstm, variant: patient_history, stage: fusion}
    model: {fusion: early}
  lstm_huber:
    experiment: {variant: patient_history, stage: loss}
    training: {loss: huber}
  lstm_derivative:
    experiment: {variant: patient_history, stage: loss}
    training: {derivative_weight: 0.1}
  lstm_spectral:
    experiment: {variant: patient_history, stage: loss}
    training: {spectral_weight: 0.01}
  lstm_residual:
    experiment: {variant: patient_history, stage: decoder}
    model: {residual: true}
  cnn_kernels:
    experiment: {architecture: cnn_lstm, variant: patient_history, stage: kernels}
    model: {kernels: [13, 7, 3]}
  cnn_pool1:
    experiment: {architecture: cnn_lstm, variant: patient_history, stage: pooling}
    model: {pool: 1}
  lstm_contrastive:
    experiment: {variant: patient_history_contrastive, stage: contrastive}
  cnn_contrastive:
    experiment: {architecture: cnn_lstm, variant: patient_history_contrastive, stage: contrastive}
  lr_01:
    experiment: {variant: patient_history, stage: learning_rate}
    training: {lr: 0.01}
  lr_0005:
    experiment: {variant: patient_history, stage: learning_rate}
    training: {lr: 0.0005}
  lr_0001:
    experiment: {variant: patient_history, stage: learning_rate}
    training: {lr: 0.0001}
  hidden_128:
    experiment: {variant: patient_history, stage: hidden}
    model: {hidden: 128}
  hidden_256:
    experiment: {variant: patient_history, stage: hidden}
    model: {hidden: 256}
  depth_2:
    experiment: {variant: patient_history, stage: depth}
    model: {depth: 2}
  depth_3:
    experiment: {variant: patient_history, stage: depth}
    model: {depth: 3}
  dropout_01:
    experiment: {variant: patient_history, stage: dropout}
    model: {dropout: 0.1}
  dropout_03:
    experiment: {variant: patient_history, stage: dropout}
    model: {dropout: 0.3}
  batch_32:
    experiment: {variant: patient_history, stage: batch}
    training: {batch_size: 32}
  weight_decay_0:
    experiment: {variant: patient_history, stage: regularization}
    training: {weight_decay: 0.0}
  scheduler_plateau:
    experiment: {variant: patient_history, stage: scheduler}
    training: {scheduler: plateau}
  embedding_8:
    experiment: {variant: patient_history, stage: embedding}
    model: {static_embedding: 8}
  cnn_lr_01:
    experiment: {architecture: cnn_lstm, variant: patient_history, stage: learning_rate}
    training: {lr: 0.01}
  cnn_lr_0005:
    experiment: {architecture: cnn_lstm, variant: patient_history, stage: learning_rate}
    training: {lr: 0.0005}
  cnn_lr_0001:
    experiment: {architecture: cnn_lstm, variant: patient_history, stage: learning_rate}
    training: {lr: 0.0001}
  cnn_hidden_128:
    experiment: {architecture: cnn_lstm, variant: patient_history, stage: hidden}
    model: {hidden: 128}
  cnn_hidden_256:
    experiment: {architecture: cnn_lstm, variant: patient_history, stage: hidden}
    model: {hidden: 256}
  context_3_1:
    experiment: {variant: patient_history, stage: context}
    data: {cache: data/research_100hz_dev_3_1, history_seconds: 3.0, horizon_seconds: 1.0, training_windows: single}
  context_5_1:
    experiment: {variant: patient_history, stage: context}
    data: {cache: data/research_100hz_dev_5_1, history_seconds: 5.0, horizon_seconds: 1.0, training_windows: single}
  context_5_3:
    experiment: {variant: patient_history, stage: context}
    data: {cache: data/research_100hz_dev_5_3_single, training_windows: single}
  context_7_3:
    experiment: {variant: patient_history, stage: context}
    data: {cache: data/research_100hz_dev_7_3, history_seconds: 7.0, horizon_seconds: 3.0, training_windows: single}
  context_8_2:
    experiment: {variant: patient_history, stage: context}
    data: {cache: data/research_100hz_dev_8_2, history_seconds: 8.0, horizon_seconds: 2.0, training_windows: single}
  aligned_3_1:
    experiment: {variant: patient_history, stage: context_aligned}
    data: {cache: data/research_100hz_dev_aligned_3_1, history_seconds: 3.0, horizon_seconds: 1.0, training_windows: single, start: 200}
  aligned_5_3:
    experiment: {variant: patient_history, stage: context_aligned}
    data: {cache: data/research_100hz_dev_aligned_5_3, history_seconds: 5.0, horizon_seconds: 3.0, training_windows: single, start: 200}

```

