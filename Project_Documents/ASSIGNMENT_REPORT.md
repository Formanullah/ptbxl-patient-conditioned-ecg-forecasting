# Contrastive Patient-Stratified ECG Forecasts

INCOMPLETE: remaining runs: cnn_lstm_demographics, cnn_lstm_ecg_only, cnn_lstm_patient_history, cnn_lstm_patient_history_contrastive, lstm_demographics, lstm_ecg_only, lstm_patient_history, lstm_patient_history_contrastive

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
yield 17,418/2,183/2,198 records and 15,023/1,942/1,904 patients.
Waveforms retain physical mV units at ingestion. No additional filtering or resampling
is applied. Standardization fits per-lead population mean/std on training INPUTS only;
both inputs and targets use that transform. Predictions are inverted to mV for metrics.

Age outside 0-120, including sentinel 300, is missing. Training-only median imputation
and population standard deviation scaling are used. Age and sex have missingness
flags; sex follows PTB-XL's male=0/female=1 coding with unknown=0.5 plus its flag.
Diagnostic SCP keys map to NORM, MI, STTC, CD and HYP. Likelihood zero means unknown
confidence, not absence. Historical indicators are the union of labels from strictly
earlier timestamps of the same patient; current/same-time labels are excluded.
Missing history is a separate flag. Available history covers 2,929 records (13.4%);
only 293 test records have it. Historical labels are a proxy, not a verified clinical
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
at 1.0 to limit exploding gradients. Maximum 25 epochs, patience 5, and minimum
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

![Dataset audit](../outputs/metadata_distributions.png)

| run | mae | rmse | mse | r2 | pearson_mean | patient_macro_mae | n_test_records | n_test_patients | epochs | best_epoch | training_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 0.132093 | 0.290380 | 0.084321 | -0.159707 | 0.408972 | 0.131781 | 2198 | 1904 | 0 | undefined | 0 |
| repeat_window | 0.155507 | 0.309010 | 0.095487 | -0.313286 | 0.354238 | 0.154881 | 2198 | 1904 | 0 | undefined | 0 |

## Observed comparisons

Lowest observed record-weighted test MAE: **persistence**, 0.132093 mV.

This is descriptive; test results were not used to tune or choose checkpoints.

## Local timing benchmark

| architecture | contrastive | records | seconds | estimated_train_epoch_seconds |
| --- | --- | --- | --- | --- |
| lstm | False | 256 | 1.329423 | 90.452674 |
| lstm | True | 256 | 4.408846 | 299.973762 |
| cnn_lstm | False | 256 | 0.794673 | 54.068779 |
| cnn_lstm | True | 256 | 2.515597 | 171.158848 |

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

### persistence

![Representative forecast](../outputs/persistence/forecast_lowest_id.png)

![Per-lead errors](../outputs/persistence/per_lead_errors.png)

### repeat_window

![Representative forecast](../outputs/repeat_window/forecast_lowest_id.png)

![Per-lead errors](../outputs/repeat_window/per_lead_errors.png)

Configs, feature manifests, histories, checkpoints, per-record errors and predictions are stored under experiments/<run>. Subgroup/horizon CSVs and remaining figures are under outputs/<run>. Re-run `python -m tsa report` to regenerate this document.
