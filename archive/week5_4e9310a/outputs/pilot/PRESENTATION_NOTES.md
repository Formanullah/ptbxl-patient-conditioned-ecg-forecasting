# Presentation speaking notes

These historical 500 Hz notes follow the 41-slide PROJECT_PRESENTATION_FEATURES.pptx. For the new protocol, use [the 100 Hz research presentation notes](../../Project_Documents/PRESENTATION_NOTES.md). The original speaking notes remain below for the historical slides.

**Feature sets:** ECG only has 0 static values; demographics has 4; patient history has 10. Contrastive training reuses the history inputs. Four variants with two architectures give eight neural runs, plus two simple baselines.

## Slide 1 - Patient-conditioned ECG forecasting

Our project predicts a short future section of a twelve-lead ECG. We compare LSTM and CNN-LSTM, first using ECG alone and then adding patient information. We also test contrastive training. The results I will present come from a small preliminary experiment, with five training epochs for each neural model.

## Slide 2 - The four assignment tasks

I will explain four stages. First, we prepare the ECG and patient information and divide the patients into training, validation and test groups. Second, we implement and train the LSTM. Third, we calculate errors and inspect the forecasts. Finally, we train CNN-LSTM under the same settings and compare the results.

## Slide 3 - The data and prediction task

We use twelve ECG leads at 500 samples per second. Three hundred samples therefore cover 0.6 seconds. The model receives that observed section and predicts the next 150 samples, or 0.3 seconds. During training we already know the actual future section, so we compare the prediction with it and use the error to improve the weights.

## Slide 4 - Prepare the ECG signals

We check each recording for the expected leads, units and valid voltage values. Then we extract the observed and future sections. We scale each lead using only the training inputs. Validation and test data use exactly the same training-based scale. This prepares comparable numerical inputs without learning preprocessing settings from the held-out data.

## Slide 5 - Prepare the patient information

We prepare age, sex and historical diagnoses separately from the waveform. Missing ages use the training median, and missingness flags tell the model which values were unavailable. Diagnoses come only from strictly earlier ECGs of the same patient. The diagnosis attached to the current ECG does not enter its forecasting inputs.

## Slide 6 - Three input sets, four experiments

We use three input feature sets. First, ECG only uses all twelve voltage leads with no static patient features. Second, demographics adds age and sex. The code represents these as four values: scaled age, an age-missing flag, encoded sex and a sex-missing flag. Third, patient history adds five prior-diagnosis indicators and a history-missing flag, giving ten static values. The five classes are NORM, MI, STTC, CD and HYP. Our fourth experiment adds contrastive training to the patient-history model, keeping the same ten static inputs. We run all four experiments with both LSTM and CNN-LSTM, giving eight neural runs. We did not train separate age-only or sex-only experiments.

## Slide 7 - Divide the patients into three groups

Training data teaches the model weights. Validation data selects the best saved checkpoint. Test data measures that selected model on unseen patients. We follow the official folds and keep each patient in only one group. Our small experiment contains 515 training ECGs, 108 validation ECGs and 116 test ECGs.

## Slide 8 - Give the data to the model in batches

The Dataset returns one example: its observed waveform, correct future waveform and selected patient features. The DataLoader combines up to 64 examples into a batch. We shuffle training examples, while validation and test order stays fixed. Both architectures use the same prepared data, so changing the architecture does not also change the cohort.

## Slide 9 - LSTM implementation: waveform and static branches

This diagram shows the patient-history model. The waveform passes through two LSTM layers and becomes a 128-value representation. The ten static patient features pass through a small linear layer and ReLU, producing 32 values. We combine those representations into 160 values. ECG-only omits the static branch, while the demographics variant supplies four static inputs instead of ten.

## Slide 10 - LSTM gates and memory flow

An LSTM keeps a cell state as memory and a hidden state as its current output. The forget gate controls how much old memory remains. The input gate controls how much new candidate content enters the memory. The output gate controls what the hidden state reveals. Training adjusts these gates through backpropagation over the sequence. We use gradient clipping to limit very large gradients.

## Slide 11 - Forecast head: one representation, 1,800 outputs

After combining ECG and patient information, the forecast head uses a 256-unit layer, ReLU and dropout. Its final linear layer produces 1,800 values because twelve leads times 150 future samples equals 1,800. We reshape these values into the future ECG. The model predicts the whole future window together. ECG-only starts this head with 128 values because there is no static branch.

## Slide 12 - Hyperparameters: model capacity and regularization

We chose two LSTM layers with hidden size 128 as a moderate starting capacity. Dropout of 0.2 adds regularization. The static embedding has 32 values, and the forecast head has 256 hidden units. These are practical starting settings. We have not performed a large search that would establish them as optimal. LSTM dropout here acts between the stacked layers.

## Slide 13 - Hyperparameters: optimizer and training budget

We train using AdamW with learning rate 0.001 and weight decay 0.0001. The batch size is 64 and the gradient norm limit is 1.0. We use seed 42 and allow five pilot epochs, with patience three. All models select their best epoch using validation forecast MSE. The short budget keeps this preliminary CPU study manageable.

## Slide 14 - One training batch: prediction, error and update

For one training batch, the model predicts the future and we calculate mean squared error against the actual future. We clear previous gradients, backpropagate the loss, clip large gradients and let AdamW update the weights. The displayed code describes plain forecasting. Validation calculates error with no weight update. Contrastive variants add their auxiliary objective during training.

## Slide 15 - Actual LSTM training history and checkpoint choice

This is the actual training log for LSTM with patient history. Validation error improved in each of the five epochs, so the selected checkpoint came from epoch five. The final validation MSE is about 0.664 in normalized units. We reached the five-epoch budget. The decreasing curve does not prove that the model had fully converged.

## Slide 16 - Recording metrics and safely resuming training

We save enough information to trace and continue each run. The configuration records its settings and features. The history CSV records the losses and epoch times. best.pt holds the checkpoint with the lowest validation error. last.pt also stores the optimizer and random states for resuming. The current training script writes a summary JSON after training.

## Slide 17 - Evaluation: load best.pt and test unseen patients

Evaluation loads the saved configuration and best checkpoint. It predicts on test patients with dropout disabled and without updating weights. We then reverse normalization so both predictions and actual voltages are in mV. Finally, the evaluation script saves numerical metrics, predictions and figures. The test answers do not select the checkpoint.

## Slide 18 - Evaluation metrics and selection rules

MAE describes average absolute voltage error. RMSE gives larger mistakes more influence. We also record R-squared and per-lead correlation, because different metrics describe different aspects of the forecast. Patient-macro MAE first averages within each patient, then gives patients equal weight. Forecast quality still needs visual inspection because low average error can hide a missed peak.

## Slide 19 - Worked example: MAE, MSE and RMSE

This small example explains the calculations. The absolute errors are 0.1, zero and 0.2 mV, so MAE is 0.1 mV. Squaring the errors gives 0.01, zero and 0.04. Their average is MSE, and taking its square root gives RMSE, about 0.129 mV. Our real evaluation performs these calculations across test recordings, all leads and future samples.

## Slide 20 - Measured LSTM test performance

For LSTM with patient history, test MAE is approximately 0.150 mV and RMSE is approximately 0.431 mV. We evaluated 116 ECGs from 100 test patients. The R-squared and correlation show that low absolute error does not imply a close match to every waveform detail. These are preliminary results from a small cohort and one training seed.

## Slide 21 - Learning curves: how to read training progress

The left panel shows training and validation forecast MSE across epochs. Both decrease in this run. These losses use normalized units, unlike the test errors in mV. The right panel is zero because this variant does not use contrastive training. Five epochs are too few to establish the final optimum or stable generalization behavior.

## Slide 22 - Forecast visualization: lowest-ID test recording

This recording was selected by the lowest test ECG identifier, independently of its error. Gray shows the observed input. After the dashed forecast boundary, the actual future and prediction appear together. We inspect all twelve leads for missed peaks, incorrect timing and amplitude differences. The ECG number identifies the recording rather than the person or a heart-rate measurement.

## Slide 23 - Forecast visualization: worst-error recording

ECG 11124 is the recording ID of the worst-error test example for this model. We selected it using the largest average absolute prediction error across leads and future samples. Its patient ID is a separate value. This deliberately difficult example shows failure behavior. We should not describe it as a typical recording.

## Slide 24 - Per-lead and horizon-wise error

The left chart shows how errors differ between leads. The right chart shows error at each position in the future window. These plots can reveal problems that an overall average hides. Every error here uses physical mV. The CSV files behind these figures let us inspect exact values, while the forecast overlays show the waveform shape.

## Slide 25 - Patient strata and uncertainty

We also calculate errors for patient groups, including age, sex, diagnosis and prior-history availability. This selected table shows some of those results and their record counts. Several groups are small, and diagnostic groups can overlap. These are descriptive observations. They do not establish a causal demographic effect or a clinical fairness conclusion.

## Slide 26 - Where evaluation metrics and figures are stored

The experiment directory contains model checkpoints, test metrics, predictions and per-record errors. The output directory contains the charts and lead, horizon and subgroup CSVs. The comparison script combines saved results, and the report script writes the assignment documents. For our completed small experiment, these files are under the pilot paths shown here.

## Slide 27 - CNN-LSTM: the purpose of the added CNN

CNN-LSTM adds a convolutional feature extractor before the LSTM. The CNN learns local waveform features, and the LSTM processes their sequence. Pooling reduces the number of recurrent steps from 300 to 150. We keep the same patient-information branch and forecast-head design. The added CNN may help, but we need the measured comparison to test that idea.

## Slide 28 - CNN operations: kernels, padding, ReLU and pooling

A convolution uses a learned local filter repeatedly across time. Our kernel lengths are seven, five and three. Padding preserves length through the convolution layers. ReLU adds a nonlinear activation. Average pooling combines adjacent feature steps and halves the sequence length. That reduces recurrent work but can lose some timing detail. Output channels are learned features, not additional recorded ECG leads.

## Slide 29 - CNN-LSTM architecture and convolution

This table traces the tensor sizes. The first convolutions change twelve input leads into 32 and then 64 feature channels. Pooling reduces 300 time positions to 150. We transpose those features for the LSTM, which still returns a 128-value summary. The final predicted ECG remains twelve leads by 150 samples, matching the original LSTM output.

## Slide 30 - Training CNN-LSTM under a matched protocol

To train CNN-LSTM, we change the model argument to cnn_lstm and keep the patient-history variant. We preserve the data, seed, optimizer settings and training budget. Validation MSE chooses its checkpoint, and evaluation uses the same test patients. This lets us compare architectures with the same available information and experimental conditions.

## Slide 31 - Learning curves: patient-history models

Here are the actual learning curves for both patient-history models. Both selected epoch five. The LSTM reached lower validation MSE in this run. These curves use standardized units, and neither five-epoch history proves convergence. The zero auxiliary panels simply show that these two particular models use plain forecasting without contrastive loss.

## Slide 32 - LSTM versus CNN-LSTM: measured test errors

Each row represents a different experiment. ECG-only has no static inputs. Demographics adds age and sex with missingness flags. Patient history adds prior diagnoses. The fourth row keeps those history inputs and adds contrastive training. Compare LSTM and CNN-LSTM within the same row. LSTM has lower observed MAE in every row for this pilot, which is not a universal architecture ranking.

## Slide 33 - Architecture comparison: errors and runtime

With the same patient-history features, LSTM has lower test MAE and RMSE than CNN-LSTM. CNN-LSTM has more parameters but completed the recorded pilot epochs faster on this CPU, consistent with its shorter recurrent sequence. The timing comes from one local run. Stored parameter totals include the projection head, which is inactive in plain forecast training.

## Slide 34 - Paired patient-bootstrap comparisons

We compare paired errors from the same test patients. We first average the recording differences within a patient, then resample patients to estimate a confidence interval. In this table, the delta is the first-named model minus its reference, so a negative delta favors the first-named model. The intervals describe variation across test patients for one fitted seed. They do not include retraining variability.

## Slide 35 - Simple baselines remain strong

Simple baselines are important reference points. Persistence repeats the last observed voltage. Repeated-window forecasting copies the last 150 observed samples. In this pilot, persistence has the lowest MAE and the repeated window has the lowest RMSE. Therefore, the trained neural models have not demonstrated an advantage over these simple forecasts under this short training budget.

## Slide 36 - Training loop and contrastive objective

Contrastive learning is an additional training objective, not a fourth input feature set. We create two slightly noisy versions of the same observed ECG and encourage their dynamic representations to be similar. Other recordings from the same patient are excluded as negatives. The future targets stay unchanged. Forecast loss still uses the clean input. Static inputs remain the same ten patient-history values.

## Slide 37 - Eight controlled experiments and two baselines

We have three input sets and four experiment variants. The fourth variant adds contrastive training to the history inputs. We train all four variants with two architectures, giving eight neural runs. The two simple baselines are additional evaluations and do not train neural weights. There are no separate age-only or sex-only runs in the current matrix.

## Slide 38 - Comparison with the supplied TFT manuscript

The supplied TFT manuscript reports the numbers shown in this table. Our LSTM values use original mV after reversing normalization. The paper uses a different scaling approach, and its split and aggregation details are not fully aligned with ours. We therefore show its results as context and do not claim a direct numerical ranking. The private PDF remains excluded from shared project files.

## Slide 39 - Literature caveats and project limitations

The paper also contains conflicting abstract and table values, so we keep that reporting caveat visible. Our study uses a short context, one fixed window, one seed and a small cohort. Only a minority of recordings have prior history. The next experiments should expand data, context and training, then repeat seeds. Clinical benefits would require a separate evaluation.

## Slide 40 - How to run and configure the project

We activate the environment once and edit the YAML configuration. Then we run preprocessing, train one chosen model and variant, and evaluate its checkpoint. We repeat training and evaluation for the comparison model. Finally, we run the baselines, comparison and report scripts. Fresh output paths keep changed experiments separate. The matching cache must exist before training.

## Slide 41 - Conclusions and references

We implemented the complete forecasting workflow and completed eight small-data neural runs plus two baselines. LSTM with patient history has the lowest neural MAE, about 0.150 mV, but persistence is better at about 0.142 mV. The project gives us a reproducible basis for larger experiments. These results do not establish clinical benefit or a general improvement from conditioning or contrastive training.

## Exact static-feature order

Demographics uses age_scaled, age_missing, sex_encoded and sex_missing. Patient history adds prior_NORM, prior_MI, prior_STTC, prior_CD, prior_HYP and history_missing. The fourth variant changes the training objective without adding input features. The waveform remains 12 leads by 300 observed samples in every variant.

Sources: src/config.py, src/preprocessing.py, src/models.py, src/train.py, src/evaluate.py, src/compare.py, configs/pilot.yaml, outputs/pilot/model_comparison.csv and the assignment report. Numerical results refer to the completed small-data pilot. The supplied reference PDF stays local.

