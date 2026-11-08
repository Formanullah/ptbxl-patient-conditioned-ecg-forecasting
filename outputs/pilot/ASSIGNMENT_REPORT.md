# Contrastive Patient-Stratified ECG Forecasts

All eight pilot neural runs and both baselines are complete.

**Small-data pilot: exploratory results, not the full PTB-XL benchmark.** Whole patients were sampled uniformly within the official folds using seed 42. All available records of each selected patient were kept. Age and waveform scalers were refitted on selected training data only. Physical voltages were recovered from the full cache by inverse transformation before refitting (float32 rounding is possible). Small cohorts and short training budgets limit conclusions and literature comparisons.

## Active cohort

| split | records | patients | with_prior_history |
| --- | --- | --- | --- |
| train | 515 | 450 | 65 |
| val | 108 | 100 | 8 |
| test | 116 | 100 | 16 |

## Research question and interpretation

Does known baseline context improve short-horizon multivariate ECG forecasting, and
does an auxiliary contrastive objective improve the learned dynamic representation?
The experiment uses age, sex and prior diagnostic superclasses. Demographic or
pathology conditioning is a hypothesis, not an assurance of improvement.

This is within-record voltage forecasting, not prediction of disease at a later visit.
No clinical false-alarm reduction or diagnostic accuracy is established by these errors.

## Data and preprocessing

PTB-XL v1.0.3: 21,799 ECGs, 18,869 patients, twelve leads. The default experiment
uses records500 at 500 Hz: samples 0:300 (0.6 s) predict 300:450 (0.3 s).
Each record contributes one fixed window. Official patient-isolated folds 1-8/9/10
yield 17,418/2,183/2,198 records in the full dataset and 15,023/1,942/1,904 patients.
Waveforms retain physical mV units at ingestion. No additional filtering or resampling
is applied. Standardization fits per-lead population mean/std on training INPUTS only;
both inputs and targets use that transform. Predictions are inverted to mV for metrics.

Age outside 0-120, including sentinel 300, is missing. Training-only median imputation
and population standard deviation scaling are used. Age and sex have missingness
flags; sex follows PTB-XL's male=0/female=1 coding with unknown=0.5 plus its flag.
Diagnostic SCP keys map to NORM, MI, STTC, CD and HYP. Likelihood zero means unknown
confidence, not absence. Historical indicators are the union of labels from strictly
earlier timestamps of the same patient; current/same-time labels are excluded.
Missing history is a separate flag. In the full dataset, available history covers 2,929 records (13.4%);
293 full-test records have it. Historical labels are a proxy, not a verified clinical
problem list. Current-record labels are used only for subgroup reporting.

## Models and hyperparameter justification

LSTM: two layers, hidden size 128; recurrent dropout 0.2. CNN-LSTM adds three padded
temporal convolutions: 12->32 (kernel 7), 32->64 (kernel 5), average pooling by 2,
64->64 (kernel 3), with ReLU. A 300-sample sequence becomes 150 CNN feature steps.
The CNN receptive field after the final convolution is 16 raw samples (32 ms at 500 Hz).
The static branch projects 0/4/10 selected features to 32 dimensions when present.
Final dynamic state and static context enter a 256-unit ReLU/dropout head and 1,800
outputs reshaped to 12x150. Every model owns a 128->128->64 projection head; it is
trained and used only for contrastive variants. Parameter counts include that head;
parameters_used_by_loss in each run distinguishes active training parameters.

Hidden 128 and two layers provide moderate capacity without an extensive sweep.
The 32-dimensional static embedding limits static-branch size. Kernels 7/5/3 capture
local voltage changes; pooling halves recurrent sequence length, with a possible cost
in timing precision. These choices are plausible starting values, not tuned optima.

AdamW learning rate 0.001 is a common initial scale; weight decay 0.0001 and dropout
0.2 regularize. Batch 64 balances memory and minibatch statistics. Clip gradient norm
at 1.0 to limit exploding gradients. Maximum 5 epochs, patience 3, and minimum
validation forecast MSE determine the checkpoint. No test-based tuning is performed.
All eight runs use seed 42, the same splits, cache, optimizer settings and output window.

For contrastive variants, two independent views augment observed inputs only, using
normalized Gaussian jitter std 0.01 and a global gain in [0.95,1.05]. No timing shifts,
polarity reversal or target augmentation occurs. The dynamic encoder supplies an
L2-normalized 64-dimensional projection. Symmetric NT-Xent uses temperature 0.1;
the paired view is positive, self and other same-patient comparisons are excluded.
Anchors without negatives are skipped and counted. The loss is normalized forecast
MSE on clean inputs + 0.1*NT-Xent. Projection happens before static fusion to limit
metadata shortcuts. Temperature and weight are declared defaults, not optimized values.

## Metric definitions

For residual e=prediction-target over records, leads and forecast samples:
MAE=mean(abs(e)) [mV], MSE=mean(e^2) [mV^2], RMSE=sqrt(MSE) [mV]. Overall
R2=1-SSE/SST with one global target mean. Pearson is calculated per lead after
concatenating records/time, then averaged over defined leads. Constant-signal cases
are null/undefined and counted. Patient-macro MAE first averages records per patient,
then patients equally. These amplitude metrics do not establish clinical morphology fidelity.

Age groups: <40, 40-59, 60-79, >=80, unknown. Sex and history availability are separate
strata. Diagnostic groups overlap because labels are multi-label. Subgroup errors
are record-weighted, with record/patient counts. Paired patient-bootstrap comparisons
average record MAE within patients, resample patients 1,000 times, and report percentile
95% intervals for B-A; negative favors B. They measure test-patient uncertainty conditional
on this fitted seed and are descriptive, without multiple-comparison correction.

## Measured results

Errors are in original mV (MSE in mV^2).

![Dataset audit](./metadata_distributions.png)

| run | mae | rmse | mse | r2 | pearson_mean | patient_macro_mae | n_test_records | n_test_patients | epochs | best_epoch | training_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_demographics | 0.155650 | 0.434724 | 0.188985 | 0.012654 | 0.202422 | 0.152838 | 116 | 100 | 5 | 5.000000 | 9.027768 |
| cnn_lstm_ecg_only | 0.154296 | 0.433948 | 0.188311 | 0.016176 | 0.200333 | 0.151576 | 116 | 100 | 5 | 4.000000 | 9.075329 |
| cnn_lstm_patient_history | 0.154251 | 0.433902 | 0.188271 | 0.016385 | 0.202367 | 0.151215 | 116 | 100 | 5 | 5.000000 | 8.763636 |
| cnn_lstm_patient_history_contrastive | 0.155719 | 0.435588 | 0.189737 | 0.008725 | 0.178517 | 0.152772 | 116 | 100 | 5 | 5.000000 | 28.105523 |
| lstm_demographics | 0.151759 | 0.431238 | 0.185966 | 0.028427 | 0.273694 | 0.148413 | 116 | 100 | 5 | 5.000000 | 18.884125 |
| lstm_ecg_only | 0.151706 | 0.430543 | 0.185368 | 0.031552 | 0.282981 | 0.148435 | 116 | 100 | 5 | 4.000000 | 15.137457 |
| lstm_patient_history | 0.150113 | 0.430582 | 0.185401 | 0.031379 | 0.288765 | 0.146684 | 116 | 100 | 5 | 5.000000 | 15.547189 |
| lstm_patient_history_contrastive | 0.151782 | 0.431546 | 0.186232 | 0.027037 | 0.284187 | 0.148438 | 116 | 100 | 5 | 4.000000 | 47.396776 |
| persistence | 0.141546 | 0.358149 | 0.128271 | 0.329855 | 0.442172 | 0.129724 | 116 | 100 | 0 | undefined | 0.000000 |
| repeat_window | 0.168642 | 0.336686 | 0.113358 | 0.407766 | 0.364880 | 0.161466 | 116 | 100 | 0 | undefined | 0.000000 |

## Observed comparisons

Lowest observed record-weighted test MAE: **persistence**, 0.141546 mV.

This is descriptive; test results were not used to tune or choose checkpoints.

Lowest observed neural-model MAE: **lstm_patient_history**, 0.150113 mV.

Its MAE is higher than persistence (0.141546 mV). This is a descriptive comparison under the configured data/training budget, not evidence of a general clinical advantage.

The pilot uses only five epochs and a small cohort. Do not infer architecture superiority or conditioning benefit from this run alone. Increase training patients and the predeclared epoch budget, preserving patient isolation and validation-based checkpoint selection.

- demographics: lstm_demographics minus lstm_ecg_only: patient-macro MAE delta -0.000022 mV, 95% interval [-0.001178, +0.001184].

- history: lstm_patient_history minus lstm_demographics: patient-macro MAE delta -0.001729 mV, 95% interval [-0.003275, -0.000242].

- contrastive: lstm_patient_history_contrastive minus lstm_patient_history: patient-macro MAE delta +0.001754 mV, 95% interval [+0.000283, +0.003351].

- demographics: cnn_lstm_demographics minus cnn_lstm_ecg_only: patient-macro MAE delta +0.001261 mV, 95% interval [-0.000025, +0.002541].

- history: cnn_lstm_patient_history minus cnn_lstm_demographics: patient-macro MAE delta -0.001623 mV, 95% interval [-0.003720, +0.000269].

- contrastive: cnn_lstm_patient_history_contrastive minus cnn_lstm_patient_history: patient-macro MAE delta +0.001557 mV, 95% interval [-0.000871, +0.003845].

- architecture: cnn_lstm_ecg_only minus lstm_ecg_only: patient-macro MAE delta +0.003141 mV, 95% interval [+0.001435, +0.005009].

- architecture: cnn_lstm_demographics minus lstm_demographics: patient-macro MAE delta +0.004425 mV, 95% interval [+0.002416, +0.006544].

- architecture: cnn_lstm_patient_history minus lstm_patient_history: patient-macro MAE delta +0.004531 mV, 95% interval [+0.002810, +0.006488].

- architecture: cnn_lstm_patient_history_contrastive minus lstm_patient_history_contrastive: patient-macro MAE delta +0.004334 mV, 95% interval [+0.002060, +0.006693].

## Local timing benchmark

Benchmark not yet run.

Benchmark estimates exclude validation, checkpoint and plotting overhead; measured run times are in the results table.

## Literature comparison

The supplied local reference is F. S. Butt et al., *Temporal Fusion Transformers for
Forecasting ECG Signals* (user-supplied 2025 manuscript), pp. 5-6, Table 1.
The PDF is not included in this repository, report bundle or Colab transfer.

| Paper experiment | MAE as reported | RMSE as reported | SMAPE as reported |
| --- | --- | --- | --- |
| 1 | 0.0604 | 0.1234 | 0.9950 |
| 2 | 0.0537 | 0.1184 | 0.9333 |
| 3 | 0.0504 | 0.1135 | 0.8791 |
| 4 | 0.0725 | 0.1466 | 1.0104 |
| 5 | 0.0674 | 0.1402 | 0.9620 |

These are contextual published values, not measured project results. The paper's abstract
states RMSE 0.061 and MAE 0.128, conflicting with Table 1; this report uses the explicit
table and does not silently correct the abstract. MAE exceeding RMSE would be inconsistent
for the same residual set under their standard definitions.

The paper reports 300 historic/150 forecast samples for experiments 1-3, age/sex static
features, robust TorchNormalizer scaling and an 80/10/10 split. Its description of
1000-to-500 Hz downsampling conflicts with PTB-XL's documented native 500 Hz/available
100 Hz recordings. Patient isolation, exact scaling/inverse scaling, lead aggregation,
and some other experiment geometries are insufficiently specified for exact reproduction.
This project uses an explicit 500 Hz protocol, train-input-only z-scores, official patient
folds, prior diagnoses and an auxiliary contrastive objective. Report our original-mV
MAE/RMSE beside those numbers with these caveats; no numerical superiority claim is justified.
SMAPE is omitted from our primary metrics because ECG voltages cross zero.

Wagner et al. (2020) defines PTB-XL, not a directly matching waveform forecast benchmark.
Strodthoff et al. (2021) reports ECG classification; classification AUC is not forecast error.
Chen et al. (2020) motivates two-view contrastive learning; the ECG adaptation here is an
experiment, not a claim that image-domain results transfer to ECG.

Sources:
- PTB-XL v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- Wagner et al.: https://doi.org/10.1038/s41597-020-0495-6
- Strodthoff et al.: https://doi.org/10.1109/JBHI.2020.3022989
- Chen et al.: https://proceedings.mlr.press/v119/chen20j.html
- Butt et al.: local user-provided manuscript; no external upload or redistribution.

## Limitations

The 0.6 s context can contain less than one beat; all results concern a 0.3 s within-record
forecast. Fixed start-zero windows cover only a small part of each ten-second recording.
One seed does not quantify optimization variability. Only a minority has prior ECG history;
historical diagnoses are annotations, not independently verified prior clinical knowledge.
Subgroups can be small and overlapping. Standardization and missingness conventions are
explicit but differ from the reference paper. MSE can produce smooth forecasts with weak
QRS peaks; inspect timing and shape visually. No diagnostic, distant-future or clinical
false-alarm claims follow from these experiments.

## Figures and provenance

### cnn_lstm_demographics

![Representative forecast](./cnn_lstm_demographics/forecast_lowest_id.png)

![Per-lead errors](./cnn_lstm_demographics/per_lead_errors.png)

![Learning curves](./cnn_lstm_demographics/learning_curves.png)

### cnn_lstm_ecg_only

![Representative forecast](./cnn_lstm_ecg_only/forecast_lowest_id.png)

![Per-lead errors](./cnn_lstm_ecg_only/per_lead_errors.png)

![Learning curves](./cnn_lstm_ecg_only/learning_curves.png)

### cnn_lstm_patient_history

![Representative forecast](./cnn_lstm_patient_history/forecast_lowest_id.png)

![Per-lead errors](./cnn_lstm_patient_history/per_lead_errors.png)

![Learning curves](./cnn_lstm_patient_history/learning_curves.png)

### cnn_lstm_patient_history_contrastive

![Representative forecast](./cnn_lstm_patient_history_contrastive/forecast_lowest_id.png)

![Per-lead errors](./cnn_lstm_patient_history_contrastive/per_lead_errors.png)

![Learning curves](./cnn_lstm_patient_history_contrastive/learning_curves.png)

### lstm_demographics

![Representative forecast](./lstm_demographics/forecast_lowest_id.png)

![Per-lead errors](./lstm_demographics/per_lead_errors.png)

![Learning curves](./lstm_demographics/learning_curves.png)

### lstm_ecg_only

![Representative forecast](./lstm_ecg_only/forecast_lowest_id.png)

![Per-lead errors](./lstm_ecg_only/per_lead_errors.png)

![Learning curves](./lstm_ecg_only/learning_curves.png)

### lstm_patient_history

![Representative forecast](./lstm_patient_history/forecast_lowest_id.png)

![Per-lead errors](./lstm_patient_history/per_lead_errors.png)

![Learning curves](./lstm_patient_history/learning_curves.png)

### lstm_patient_history_contrastive

![Representative forecast](./lstm_patient_history_contrastive/forecast_lowest_id.png)

![Per-lead errors](./lstm_patient_history_contrastive/per_lead_errors.png)

![Learning curves](./lstm_patient_history_contrastive/learning_curves.png)

### persistence

![Representative forecast](./persistence/forecast_lowest_id.png)

![Per-lead errors](./persistence/per_lead_errors.png)

### repeat_window

![Representative forecast](./repeat_window/forecast_lowest_id.png)

![Per-lead errors](./repeat_window/per_lead_errors.png)

Configs, feature manifests, histories, checkpoints, per-record errors and predictions are stored under experiments/<run>. Subgroup/horizon CSVs and remaining figures are under the configured outputs directory. Re-run `python src/report.py --config configs/pilot.yaml` to regenerate.

