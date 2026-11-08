# Presentation notes: current project

Use the active assignment report for current patient counts and measured values. Do not copy scores from historical slides into the larger-cohort results. The expanded candidate passed 60 software tests, its real-cache audit, and training/inference batch checks. Expanded-cohort neural training/evaluation remain pending.

## 1. Research question

“We test whether patient context helps predict future ECG voltages. We observe five seconds at 100 Hz and predict the next three seconds across all twelve leads. Age, sex and strictly prior diagnostic history can condition the forecast.”

## 2. Data and preprocessing

“Our main configuration selects 4,500 training, 600 validation and 100 reserved test patients. Whole patients stay within official folds. There are 15,579 training windows from 5,193 recordings, 677 validation recordings and 114 reserved test recordings. Validation and test use one window per recording. Scaling and imputation fit training data only. Current-record diagnoses never become historical inputs.”

“Native 100 Hz reduces the sequence length compared with 500 Hz, making longer observed context practical. It also has coarser temporal resolution; it is not a claim of better clinical signal quality. Five hundred observed samples at 100 Hz represent five seconds.”

“The current input uses no extra filter and training median/IQR scaling. Causal high-pass and bandpass options remain available for controlled comparisons. The future target remains original unfiltered voltage. Filtering is an experiment, not assumed improvement.”

## 3. Features

“ECG-only has zero static inputs. Demographics has four values: age, sex and flags. Prior history has ten values, adding five diagnostic superclasses and a history flag. Expanded features total 22 with height, weight, BMI, observed heart rate and RR summaries. Missing features receive training-fitted imputation and explicit flags.”

“mRMR balances training-target relevance against feature redundancy. Grouped permutation measures validation error changes when related features and flags are shuffled together. These rankings describe predictive association, not causality.”

## 4. LSTM and CNN-LSTM

```text
LSTM:      ECG sequence → LSTM memory → dynamic state ─┐
                                                     ├→ forecast head → [B,300,12]
Late fusion: static features → embedding ─────────────┘

CNN-LSTM:  ECG → Conv k7 → Conv k5 → AvgPool /2 → Conv k3 → LSTM → head
Early fusion: concatenate the repeated static embedding before the temporal encoder.
```

“The LSTM input, forget and output gates control its memory. At each timestep it updates the cell state and hidden state. CNN layers learn local temporal patterns; the LSTM then models their sequence. Early fusion lets context influence the encoder; late fusion joins it at the head.”

For an LSTM step, with `z_t = [h_(t−1), x_t]`:

```text
f_t = sigmoid(W_f z_t + b_f)       forget gate
 i_t = sigmoid(W_i z_t + b_i)      input gate
 g_t = tanh(W_g z_t + b_g)         candidate memory
 c_t = f_t * c_(t−1) + i_t * g_t   cell update
 o_t = sigmoid(W_o z_t + b_o)      output gate
 h_t = o_t * tanh(c_t)            hidden state
```

“Backpropagation through time computes gradients through recurrent operations. AdamW updates weights, and gradient clipping limits large updates. The future target is hidden from the input but used to calculate the training loss.”

## 5. Training and comparisons

“The updated candidate uses hidden size 128, two LSTM layers, dropout 0.2, a 128-unit forecast head, batch size 64 and learning rate 0.0005. Weight decay is 0.0005. Training has a maximum of 60 epochs, a plateau learning-rate scheduler, and early stopping after ten non-improving epochs. The larger recurrent encoder and smaller head reduce total parameter count. The previous models selected epochs 12 and 25; extra epochs worsened validation error. The new settings still need measured validation.”

“We compare the same architecture with different feature sets and compare LSTM versus CNN-LSTM with the same features, patients and windows. Contrastive learning adds agreement between two augmented observed views. It excludes same-patient negatives and does not use future ECG as an input.”

## 6. Evaluation and results

“Best checkpoints are selected by validation patient-macro MAE in mV. We also report RMSE, MSE, R², per-lead Pearson, subgroup errors and QRS measures. A constant prediction can have low MAE while missing every heartbeat, so waveform overlays and peak scores matter.”

“For the current larger cohort, training and evaluation remain pending. Present only values actually populated in the active report. Earlier small-cohort neural predictions did not preserve future R-peaks reliably; more data is a testable next step, not a guaranteed fix.”

## 7. Reproducibility and next steps

“One main YAML controls the cohort and training. Optional named recipes inherit it. Dataset-specific paths and provenance checks keep scores separate. We manually prepare, audit, train, evaluate and regenerate the report. Full fold-10 evaluation follows frozen full-population validation choices.”

The earlier presentation and report are historical artifacts under `outputs/pilot/`. Superseded files are backed up locally; no private paper is included in transfers.
