# Patient-conditioned ECG forecasting: assignment report

Generated from saved artifacts at 2026-11-08T23:32:19+00:00 (UTC).

## Current configuration and execution state

Requested cohort: **train: 4500 patients, val: 600 patients, test: 100 patients**

Protocol: **100 Hz**, **5 seconds observed → 3 seconds forecast**, twelve leads. Input/target lengths: **500 / 300 samples**; layout: `time_first`.

Preprocessing: `none` filter; `robust` scaling. Seed: 42. Configured maximum: 60 epochs. Early stopping: True.

Cache: `data/custom_100hz_5s_3s_b088aaf9/6ba08ad4`. Experiments: `experiments/custom_100hz_5s_3s_b088aaf9`. [Exact report configuration](../outputs/custom_100hz_5s_3s_b088aaf9/report_config.json).

**Data prepared.** Actual cohort counts:

| split | patients | records | windows |
| --- | --- | --- | --- |
| train | 4500 | 5193 | 15579 |
| val | 600 | 677 | 677 |
| test | 100 | 114 | 114 |

Manifest fingerprint: `45e071a2fd98ed4ebfb1b2af3c41b0b3f7a45ca2d95f9a29dca370b2a7a64554`. Preparation took 43.3 recorded seconds.

Completed comparable **val evaluations: 1**. No measurements from a different cohort/window protocol are inserted into this report.

| run | state |
| --- | --- |
| cnn_lstm_patient_history | val evaluated |

## Methods and leakage controls

Official folds 1–8 train, fold 9 validates, and fold 10 is reserved for explicit final evaluation. Selection is by whole patient, retaining their eligible recordings. Training alone is shuffled. Training can use overlapping windows; validation/test use one configured window per recording.

Waveform scaling, age imputation and expanded-feature transforms fit training inputs only. The age-300 sentinel is missing. Sex has an unknown encoding and a missingness flag. Historical diagnoses use strictly earlier timestamps; current/equal-time labels are excluded. Heart-rate and RR features see observed samples only. Causal filters operate on the observed window; future targets remain unfiltered physical voltages, affine-scaled for training.

Feature choices: ECG-only (0 static values), demographics (4), prior history (10), expanded (22), or groups chosen by training-only mRMR / validation grouped permutation importance. Missingness flags remain with their features. All models retain twelve ECG leads.

LSTM summarizes the observed sequence with recurrent gates. CNN-LSTM first applies three temporal convolutions and average pooling, then an LSTM. Late fusion joins the temporal state and static embedding at the forecast head; early fusion broadcasts the embedding before the temporal encoder. Contrastive learning uses observed-only augmented views and excludes same-patient negatives.

Training uses AdamW and gradient clipping. Checkpoints are selected using validation patient-macro MAE in mV. The supplied settings are practical development choices, not proven optima for every cohort. best.pt and last.pt have different purposes: selected inference weights versus resumable final state.

## Metrics and measured comparison

MAE/RMSE are mV; MSE is mV². Patient-macro MAE averages within patients, then equally across patients. R² uses one global target mean. Pearson is computed per lead; undefined values remain null. QRS scores compare WFDB XQRS detections on lead II with one-to-one matching within the configured tolerance. They are detector proxies, not expert annotations; timing/amplitude results can be affected by misplaced detections. Clinical usefulness is not established by voltage error.

| run | model | variant | patient_macro_mae | mae | rmse | mse | r2 | pearson_mean | qrs_f1 | peak_timing_mae_ms | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_patient_history | cnn_lstm | patient_history | 0.103743 | 0.104373 | 0.214096 | 0.045837 | 0.056852 | 0.255654 | 0.010695 | 23.125000 | 764704 |

## Executed settings and budgets

| run | filter | scaling | fusion | hidden | depth | dropout | batch_size | lr | epoch_budget | actual_epochs | best_epoch | final_val_patient_macro_mae | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_lstm_patient_history | none | robust | late | 128 | 2 | 0.200000 | 64 | 0.000500 | 60 | 60 | 35 | 0.104124 | 4290.172204 |

## Findings from this cohort

- Lowest measured neural validation patient-macro MAE: **cnn_lstm_patient_history**, **0.103743 mV**. This is a descriptive validation minimum, not a global optimum.

## Paired uncertainty

Paired bootstrap resamples patients; differences are B minus A. Negative favors B. These are descriptive intervals after validation selection, with one seed and no multiplicity correction. Check the saved run settings before interpreting a difference as an isolated architecture/feature effect.

No completed measurements for this configuration.

## Figures and detailed artifacts

![Current cohort comparison](../outputs/custom_100hz_5s_3s_b088aaf9/val_comparison.png)

- **cnn_lstm_patient_history**: [evaluation artifacts](../outputs/custom_100hz_5s_3s_b088aaf9/cnn_lstm_patient_history/val).

![cnn_lstm_patient_history: learning_curves](../outputs/custom_100hz_5s_3s_b088aaf9/cnn_lstm_patient_history/learning_curves.png)


![cnn_lstm_patient_history: forecast_median_error](../outputs/custom_100hz_5s_3s_b088aaf9/cnn_lstm_patient_history/val/forecast_median_error.png)

## Limitations and next actions

Inspect waveform and QRS plots alongside scalar errors. One seed and repeated validation choices limit conclusions. Overlapping training windows are correlated. Prior-record labels are a proxy for clinically available history. Historical 500 Hz/short-horizon results and the earlier tiny 100 Hz development cohort use different protocols/populations and are not copied into this comparison.

After changing dataset settings, prepare and audit that cohort, train each desired model, evaluate on validation, and regenerate this report. Reporting refreshes only when explicitly requested. For final testing, use full mode, select with full validation, freeze checkpoint hashes and explicitly enable test access. See [the run guide](CONFIGURATION_AND_RUN_GUIDE.md).

## References

- [PTB-XL v1.0.3 and official fold guidance](https://physionet.org/content/ptb-xl/1.0.3/)
- [SciPy causal SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html)
- [WFDB processing and XQRS](https://wfdb.readthedocs.io/en/latest/processing.html)
- [Peng et al., mRMR (2005)](https://doi.org/10.1109/TPAMI.2005.159)
- Private-manuscript literature values remain in the archived original assignment report. Different splits, scaling, sampling rates and horizons prevent direct numerical superiority claims.
