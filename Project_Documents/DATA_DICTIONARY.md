# Data dictionary

The primary protocol is PTB-XL v1.0.3 native records100, 100 Hz, twelve leads. Old 500 Hz caches use their original schema and remain separate.

## Waveforms and identifiers

| Field | Meaning |
| --- | --- |
| `X_<split>.npy` | Observed input after optional causal filtering and training-fitted affine scaling; disk layout `[windows,12,history]` |
| `y_<split>.npy` | Original unfiltered future voltage after the same affine scaling; disk layout `[windows,12,horizon]` |
| `raw_X_<split>.npy` | Observed input in original mV, used for baseline and morphology evaluation |
| `static_<split>.npy` | All 22 candidate static/engineered fields in the feature order below |
| `ecg_id` | Recording identifier, not a heart-rate value or patient ID |
| `patient_id` | Person identifier; never supplied as a model identity feature |
| `window_start` | Observed-window start in raw samples |
| `window_id` | Unique `<ecg_id>:<window_start>` string |
| `split` | Official-fold membership: train 1-8, validation 9, test 10 |
| `history_available` | An earlier valid-timestamp ECG exists for this patient |
| `NORM,MI,STTC,CD,HYP` in metadata | Current-record superclass flags, for evaluation only |

The default Dataset returns time-first `[500,12]` inputs and `[300,12]` targets. Set `data.layout: channels_first` for `[12,500]` and `[12,300]`; the model and evaluation handle either layout. Changing layout changes protocol identity.

## Static/engineered columns

| Order | Fields | Preparation |
| --- | --- | --- |
| 0-1 | `age_scaled`, `age_missing` | Age outside 0-120, including 300, is missing; training-record median imputation and observed-training-age std |
| 2-3 | `sex_encoded`, `sex_missing` | 0 male, 1 female, 0.5 unknown; separate missing indicator |
| 4-8 | `prior_NORM`, `prior_MI`, `prior_STTC`, `prior_CD`, `prior_HYP` | Union from strictly earlier same-patient timestamps; same-time/current labels excluded |
| 9 | `history_missing` | 1 when no earlier valid-date recording is available |
| 10-11 | `height`, `height_missing` | Centimetres; accept 50-250, otherwise missing |
| 12-13 | `weight`, `weight_missing` | Kilograms; accept 10-400, otherwise missing |
| 14-15 | `bmi`, `bmi_missing` | weight / (height in metres)^2; missing if either input missing |
| 16-17 | `heart_rate`, `heart_rate_missing` | 60 / mean RR seconds from observed lead-II XQRS peaks; requires at least two detected peaks |
| 18-19 | `rr_mean`, `rr_mean_missing` | Observed mean interpeak interval, seconds |
| 20-21 | `rr_std`, `rr_std_missing` | Population std of observed RR, seconds; requires at least two intervals |

Expanded continuous columns are median-imputed and standardized using selected training windows only; their missing flags are unscaled. The height/weight ranges are declared plausibility rules, not a clinically validated eligibility rule. All-missing/constant training columns receive a stable fallback median/scale. Missing dates never create historical information. SCP likelihood zero means unknown confidence, so a diagnostic code's presence is retained.

## Feature variants

- `ecg_only`: no static values.
- `demographics`: first four columns.
- `patient_history` / `patient_history_contrastive`: first ten columns; the latter adds a training loss, not extra input fields.
- `expanded`: all 22 columns.
- `mrmr` / `permutation`: ordered named columns from a fingerprint-checked selection artifact. Missingness flags stay with their associated groups.

## Manifest and metric units

`mean_mV` and `std_mV` are legacy-compatible names for affine center/scale. For robust scaling they mean median and IQR, not mean and standard deviation. Inverse transformation is `mV = normalized * std_mV + mean_mV`. It does not reverse filtering. The manifest includes patient selections, patient/record/window counts, source-metadata hashes, output-array hashes, feature statistics and protocol fingerprint.

MAE/RMSE are mV, MSE is mV squared, peak timing is milliseconds, peak amplitude error is mV. QRS precision/recall/F1 are detector-based ratios. Undefined metrics are JSON null, not fabricated zeros. Residuals are prediction minus actual. Paired-bootstrap deltas are model B minus model A.
