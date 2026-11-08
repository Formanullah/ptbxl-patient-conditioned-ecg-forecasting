# Patient-conditioned ECG forecasting — current assignment report

## Current status

This document is a manually updated status snapshot. The configuration-driven report generator has been edited but **has not been run after the cleanup**, following your instruction not to execute code. Model performance for the larger cohort is pending.

## Selected protocol and dataset

- PTB-XL native 100 Hz; all twelve leads.
- Five seconds observed → three seconds predicted: 500 input samples and 300 target samples.
- Whole-patient selection within official folds 1–8 / 9 / 10.
- Main configuration: `configs/training.yaml`.
- Requested patients: 1,500 train / 300 validation / 100 reserved test.
- No additional input filter; training-input median/IQR scaling.

The cohort was prepared before the stop-execution instruction:

| Split | Patients | Recordings | Forecast windows |
| --- | ---: | ---: | ---: |
| Training | 1,500 | 1,731 | 5,193 |
| Validation | 300 | 336 | 336 |
| Reserved test | 100 | 114 | 114 |

Training uses three valid windows per recording with a one-second stride. Validation/test each use one configured window per recording. The reserved test subset has not been evaluated.

## Execution state

| Stage | State |
| --- | --- |
| Larger cohort preprocessing | Completed before the no-execution instruction |
| Leakage audit for the larger cohort | Pending |
| Tests/lint after cleanup | Not run, as requested |
| LSTM/CNN-LSTM training for this cohort | Pending |
| Validation waveform/QRS evaluation | Pending |
| Baseline evaluation for this cohort | Pending |
| Final full-population testing | Pending; the current cohort is a development subset |

No previous small-cohort score is presented as a score for this larger cohort.

## Methods and configuration

Models compare ECG-only, demographics, strictly prior diagnostic history and optional contrastive learning. Expanded-feature and feature-selection comparisons are optional. Age imputation, waveform scaling and feature statistics use training records/windows only. Age 300 is missing; missingness flags remain explicit. Current/equal-time diagnoses are excluded from history. Observed ECG alone supplies heart-rate and RR features.

The current model settings are hidden size 64, one recurrent layer, dropout 0.2, static embedding 16 and a 256-unit forecast head. CNN-LSTM uses temporal kernels 7/5/3 and average pooling by two. AdamW uses learning rate 0.001, weight decay 0.0001, batch size 64 and gradient clipping 1.0. The configured budget is 60 epochs with early stopping disabled. These are development starting choices; no larger-cohort optimum has been established.

The best checkpoint is selected by validation patient-macro MAE in original mV. Evaluation will report voltage errors, R², per-lead Pearson, subgroup and horizon errors, and detector-based peak measures. QRS locations are algorithmic estimates, not cardiologist annotations. Low MAE alone does not establish useful cardiac forecasting.

## How this report updates

When you choose to run the project:

1. Run the tests, preprocessing and leakage audit.
2. Train each desired model manually.
3. Evaluate it explicitly on validation patients and run the baselines.
4. Run `python src/report.py`.

The generator reads the selected configuration, actual cache manifest, saved run settings, histories and evaluation files. It replaces this active report with the matching cohort's measured tables, findings, confidence intervals and figures. A changed cohort or window geometry receives separate cache/result paths. Missing measurements remain pending; reporting does not train or run inference.

Merely changing YAML does not update a Markdown document. Report generation is an explicit manual step. See the [run guide](CONFIGURATION_AND_RUN_GUIDE.md).

## Historical evidence, kept separate

The original 500 Hz assignment is preserved in [its result folder](../outputs/pilot/ASSIGNMENT_REPORT.md). The completed tiny 100 Hz development study is preserved in [its historical report](../outputs/research_100hz_dev/RESEARCH_REPORT.md). Their data populations and/or forecasting geometry differ from the current configuration.

The earlier neural forecasts often had low-amplitude outputs and poor R-peak preservation; a constant training-median baseline had lower MAE. Increasing data size is a new experiment, not a guarantee this limitation will disappear. The raw-data audit previously checked all 21,799 native 100 Hz recordings without invalid records. That audit and the earlier 52-test pass are historical evidence, not validation of the latest refactor.

## References

- [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
- [SciPy SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html)
- [WFDB XQRS processing](https://wfdb.readthedocs.io/en/latest/processing.html)

The private manuscript remains local and excluded from repository/Colab transfers. Its short literature comparison remains in the historical assignment report; different protocols prevent a direct numerical superiority claim.
