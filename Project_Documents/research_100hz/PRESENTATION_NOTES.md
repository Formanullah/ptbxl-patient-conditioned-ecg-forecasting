# Presentation notes: the 100 Hz research study

These speaking notes describe the new protocol. Existing PowerPoints describe the historical 500 Hz assignment and have not been presented as new experimental evidence.

## Question and data

“We test whether preprocessing, patient context, feature selection and model changes improve ECG forecasting. We observe five seconds at native 100 Hz and predict the next three seconds for all twelve leads. We select methods with validation patients and reserve official fold 10.”

“Our CPU development study has 64 training patients with 76 ECGs and 24 validation patients with 25 ECGs. Three training windows per recording produce 228 examples. Validation uses one window per recording. The reserved small test subset is not used for method selection. Training inputs alone determine scaling and imputation.”

## Features and leakage

“ECG-only has no static inputs. Demographics adds four values: age, sex and their missingness flags. Prior history adds five superclass indicators and a history flag, giving ten values. Expanded inputs add height, weight, BMI, observed-only heart rate and RR summaries with flags, giving 22 values. Current and equal-time diagnoses never enter historical inputs. We keep all twelve ECG leads.”

“Automated checks verify patient, record and window isolation, train-only fitting, chronology, future-mutation invariance, feature-selection provenance and checkpoint selection. The cache guard caught an inherited-path mismatch before any arrays were overwritten. That configuration-resolution problem was corrected and regression-tested.”

## Preprocessing

“P0 uses no additional filter and z-scores. P1 uses a causal 0.5 Hz high-pass filter. P2 uses a causal 0.5-40 Hz bandpass. P3 uses no filter and training median/IQR scaling. Filters see observed samples only, and future targets remain original voltages. We retain and inspect causal phase delay and boundary behavior.”

“Robust scaling improves validation patient-macro MAE from about 0.11010 to 0.10793 mV, but slightly worsens RMSE. Neither filter improves the primary metric. Combining the selected filter and scaler gives the same configuration as P3. We reuse that measurement rather than pretending another model was trained.”

## Feature selection

“mRMR balances relevance against redundancy using training-only future-waveform summary targets. It selected age, height and mean RR, including flags, for six values. Grouped permutation measures validation error after jointly shuffling related features and flags between patient donors. It selected age and mean RR, giving four values. Its importances describe predictive association, not causality. Both methods change error only slightly.”

## Architectures and training

“LSTM summarizes the observed sequence with recurrent memory. CNN-LSTM first learns local features using convolution and then processes their sequence with an LSTM. Late fusion adds patient context after the temporal encoder. Early fusion repeats the context at each observed step before the LSTM or CNN. The measured early-fusion differences are small.”

“We screen four learning rates, hidden sizes 64/128/256, depths one to three, dropout 0.1/0.2/0.3 and batches 32/64. Other controlled runs vary kernels, pooling, weight decay, scheduler, embedding and loss. This is a staged search, not an exhaustive grid. Every result links to its configuration and checkpoint.”

“The objective is validation patient-macro MAE in original mV. Matched conditioned LSTM screening nominally favors learning rate 0.0005, and CNN screening 0.0001, but differences are tiny. Within the declared 0.0005 mV near-equivalence tolerance, simpler ECG-only hidden-64/depth-one candidates remain reasonable. Their retained configuration uses dropout 0.2, batch 64 and learning rate 0.001. These are development choices, not universal optima.”

## Complete epoch budgets

“We execute every epoch in the 60- and 100-epoch runs, with early stopping disabled. best.pt stores the minimum validation-MAE checkpoint; last.pt keeps final weights and resumable state. Longer training retains early best checkpoints without improving the selected validation score. Final-epoch performance can be worse, so it is documented separately.”

## Evaluation and QRS

“MAE measures average absolute voltage error. RMSE gives larger mistakes more influence. Patient-macro MAE gives patients equal weight after averaging their recordings. We also inspect twelve-lead overlays, correlation, per-lead and horizon errors.”

“WFDB XQRS detects lead-II peaks in actual and predicted futures, using the same observed prefix. One-to-one matching allows 100 ms. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is 2TP/(2TP+FP+FN). Timing and amplitude errors use matched peaks. No-peak/no-peak windows have undefined scores, not perfect scores. Detector-derived peaks are not expert clinical annotations.”

## Actual findings

“Selected CNN-LSTM validation patient-macro MAE is about 0.10785 mV. Persistence is about 0.16271 and repeated-window forecasting about 0.15753. However, a constant forecast using each lead's training median is lower, about 0.10771.”

“The selected neural forecasts match no detector-derived future R-peaks. Repeated-window forecasting has detector F1 around 0.430. Lower voltage error therefore does not establish useful morphology prediction. Extra complexity, contrastive loss and prolonged training do not show a robust benefit here. The residual forecast head is substantially worse and remains in the table.”

## Longer observation controls

“We align targets when comparing history lengths. Three and five seconds of history both predict seconds five to six. Five and seven seconds of history both predict seconds seven to ten. All use one training window per record. Start-zero comparisons with different future intervals remain descriptive. Only the aligned pairs can address whether extra history helps on the same future waveform.”

## Limits and remaining work

“One seed and repeated comparisons on 24 validation patients can overfit selection. Prior annotations are a proxy for clinically available history. Detector assumptions, short windows and amplitude thresholds limit morphology conclusions. We have not established clinical false-alarm reduction.”

“Full-data recipes and exact pending commands are supplied, but official full-data model training and fold-10 testing remain pending on this CPU-only host. We must select on full fold nine, freeze checkpoint hashes and only then explicitly evaluate fold ten. Development validation scores are not final test results.”

## Figures to show

Show preprocessing MAE/RMSE comparisons, PSD/STFT/QRS filter examples, feature rankings, 60/100-epoch learning curves, and the QRS overlays for a neural model and repeated-window baseline. Explain unsuccessful morphology alongside the voltage metrics. The supplied TFT PDF remains local; different protocols prevent a direct numerical ranking against this study.
