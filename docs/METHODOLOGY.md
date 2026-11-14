# Methodology

## Task and data

The primary task is within-record waveform forecasting: observe samples `0:700`
and predict `700:1000` in every valid 100-Hz PTB-XL ECG. All 12 leads are retained.
This is not prediction of later clinical events or a longitudinal patient outcome.
The separate 500/500 option is a 5s-to-5s ablation; its results cannot be ranked
directly against 7s-to-3s scores.

[PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) contains 21,799 ECGs
from 18,869 patients. Local inspection confirmed matching counts, complete native
100-Hz WFDB pairs, and zero cross-fold patients. Official folds 1–8 contain
17,418 records / 15,023 patients; fold 9 contains 2,183 / 1,942; fold 10 contains
2,198 / 1,904. These describe the source dataset, not completed model experiments.
See [Wagner et al.](https://doi.org/10.1038/s41597-020-0495-6) for the original
publication; its older release counts should not replace v1.0.3 counts.

Patients, not records, are sampled deterministically within official folds.
Every eligible record of each selected patient is retained. There is exactly one
forecast window per record. The local pilot selects 300 training / 75 validation
patients; medium screening selects 1,500 training patients and all validation
patients; full mode uses every official training/validation patient.

WFDB reads physical mV. Preparation validates checksums, 1,000×12 shape, 100 Hz,
standard lead order and finite values. Invalid records stop the stage and are
listed explicitly. Checksummed raw memmaps are CPU/disk caches, not GPU-resident
datasets. Base manifests and exact record/patient IDs persist with the results.

## Matched preprocessing

P0 uses raw observed history. Fit the per-lead population mean and standard
deviation using raw training history only. Transform histories and training
targets with these fixed statistics; report physical-mV errors.

P1 applies `scipy.signal.sosfilt` to observed history only, using a Butterworth
0.5–40 Hz bandpass. Prototype N=4 gives an eighth-order bandpass, represented by
four second-order sections. Initial conditions use `sosfilt_zi` multiplied by the
first observed value for each lead. Reset state between records. P1 uses exactly
the same raw-input scaler and raw future target as P0. No `filtfilt`, target
filtering, or whole-record look-ahead is used. See [SciPy butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
and [sosfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html).

Filtering is a hypothesis. A 0.5-Hz high-pass can alter ST morphology; low-pass
filtering can soften QRS detail; causal IIR filtering changes phase and has edge
transients. Native 100-Hz signals cannot reproduce all high-frequency ECG content.
The [AHA/ACCF/HRS statement](https://doi.org/10.1016/j.jacc.2007.01.024) concerns
ECG fidelity; these research settings are not a claim of diagnostic-standard
signal preservation. Raw/filtered waveforms, history-only distortion, Welch PSDs,
and an observed-amplitude-centred zoom document changes without fabricated peaks.

## Models and features

Both model classes predict the whole future directly. LSTM uses one recurrent
layer, hidden size 64, and a 128-unit ReLU/dropout forecast head initially.
CNN-LSTM first uses Conv1d kernels 7 and 5, channels 32 and 64, same padding,
ReLU, then average pooling by 2. A 700-step history becomes 350 recurrent steps.
The convolution can mix neighboring observed samples, but never target samples.

F0 has no static inputs. F1 has age, sex, and missingness flags; age scaling and
imputation fit training records only. A 16-value learned static embedding is
broadcast before the LSTM or first convolution. Only active forward-path
parameters are counted. No auxiliary projector, attention, or contrastive loss
exists in the primary classes.

F2 is disabled by default. If enabled, annotations must come from strictly
earlier records of the same patient. Equal or missing timestamps contribute no
prior record. These annotations are proxies rather than verified clinical
problem lists. Dataset date shifts preserve ordering within patients, not timing
comparisons between patients.

The required feature-selection extension runs after the primary comparison. Its
expanded pool includes height, weight, BMI, observed-history HR/RR estimates, and
their flags; audited prior diagnosis history may be added. These are separate
experiments, not changes to the F0/F1 definition. Train-only grouped mRMR uses
24 future RMS/difference-RMS summaries from training targets; no held-out outcomes
are used for ranking. Grouped permutation uses validation patient-macro error,
five repeats, patient donors and joint movement of variables/flags. The first
donor record approximates within-patient variation, which limits interpretation.
Ranks are predictive associations, not causal effects. All 12 ECG leads remain.

## Training and selection

Use seed 42, batch size 64, normalized MSE, AdamW, weight decay 1e-4, clipping
norm 1.0 and initial LR 5e-4. Medium screening compares four ECG-only model/filter
conditions for 60 epochs, then F0/F1 under each model's selected filter. Tune
LRs 1e-4, 5e-4 and 1e-3 with other settings fixed; hidden size 128 is optional.
Select by validation patient-macro MAE, then RMSE, active parameters and runtime.
No early stopping truncates the 60/120-epoch budgets.

Full mode starts four fresh runs: both architectures with P0 and P1, using the
same selected features/settings within each architecture's filter pair, for
120 epochs on full official train/validation data. Save epoch-60/120 checkpoints
and best-through-milestone validation evidence. Compare learned recipes as
recipes if different architectures have different selected LRs; isolate an
architecture effect only in otherwise matched controls.

Checkpoint every epoch. Resume restores weights, optimizer, loader/Python/NumPy/
PyTorch RNG and scheduler state (None for these fixed-LR runs). An interrupted
epoch is replayed. Configuration, source, cohort, preprocessing, feature and
scaler mismatches require a distinct experiment ID. Versions/devices are recorded;
bitwise agreement across different hardware/software is not promised.

## Final test and reporting

Freeze all four selected checkpoint hashes, configuration hashes, validation
evidence and raw scaler before enabling fold 10. The final ledger verifies and
reuses completed conditions; interrupted ones retry the same frozen condition.
Persistence repeats the last raw historical voltage. Repeat-window copies the
last 300 observed samples for the same 300-sample target. ARIMA is deferred.

Metrics are original-mV MAE/RMSE, MSE in mV², global R², per-lead Pearson and
patient-macro MAE. Average each patient's record errors before averaging patients
for selection. Undefined correlations remain null. Paired intervals resample
patients, not individual time points, and are descriptive after validation-based
selection, with one seed and no multiplicity correction.

Reports exclude synthetic/pilot diagnostics from research rankings. Missing or
incompatible artifacts remain pending/unavailable. A pre-specified ID example
and a labelled median-error example show history, future truth and forecast for
all twelve leads. Good scalar voltage error does not establish heartbeat fidelity
or clinical usefulness. No full-data GPU performance is claimed until run artifacts
prove completion.
