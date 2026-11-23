# Week 5 measured validation results

These are **real PTB-XL v1.0.3 CPU results**, limited to a four-epoch development
experiment. They are not a final test result or a ranking for the full dataset.
Run the [Week 5 notebook](../notebooks/Week5_ECG_Forecasting.ipynb) to reproduce
the workflow. Saved local metrics, histories, checkpoints and plots are in
`results/week5_cpu_450x100_e4/pilot/`.

## 1. Dataset and number of patients

The [official release](https://physionet.org/content/ptb-xl/1.0.3/) contains 21,799
ECGs from 18,869 patients. The primary runs used 515 training ECGs from 450 patients,
and 110 validation ECGs from 100 patients. Each example used the native 100 Hz
twelve-lead recording, with 700 observed and 300 target samples per lead.

## 2. Patient-safe fold validation

Official folds 1–8 supplied training, and fold 9 supplied validation. No patient
or ECG ID appeared across partitions. Fold 10 remained unopened in this real run.

## 3. Missing data and preprocessing

The notebook reports metadata missingness and invalid ages, and checks shape,
sample rate, lead order, units, and finite values before caching selected signals.
In the 19,601 development ECG rows, 259 ages fell outside the accepted
0–120 range; 67.99% of height entries and 57.42% of weight entries were missing.
P0 and P1 share the scaler fitted on raw training history. P1 applies a causal
0.5–40 Hz Butterworth bandpass to observed samples only. Both predict the same raw
future; any difference is attributable to the compared pipeline settings, not a
filtered validation target. For optional F1, the training median/mode fill missing
age/sex and the model also receives missingness indicators.

## 4. LSTM architecture

The ECG-only LSTM used two recurrent layers, 128 hidden units and a direct forecast
head. It has 685,712 trainable parameters.

## 5. CNN-LSTM architecture

Two convolution blocks and average pooling transform 700 observed steps into a
350-step feature sequence for the LSTM. The ECG-only model has 725,360 trainable
parameters and predicts the same 300×12 future.

## 6. Training and convergence

The four primary conditions used seed 42, batch size 64, AdamW at learning rate
0.001, dropout 0.2 and four epochs. All four reached their lowest recorded
validation patient MAE at epoch 4, the end of this short run. This does not show
convergence at a higher budget.

Two additional runs used E1 with the same 100 validation patients:

| Change | Before (mV) | After (mV) | Relative reduction | Paired 95% change interval (mV) |
|---|---:|---:|---:|---:|
| 225 → 450 training patients, 4 epochs | 0.117274 | 0.117035 | 0.204% | −0.000371 to −0.000081 |
| 2 → 4 epochs, 450 training patients | 0.117196 | 0.117035 | 0.138% | −0.000287 to −0.000007 |

Both changes reduced validation error in this run. A larger cohort also gave
the optimizer more batches per epoch; the cohort comparison does not isolate
total update count. Best-epoch selection uses these same validation ECGs, so
the budget comparison is descriptive and may be optimistic. No comparison
establishes an improvement for a full cohort or an unseen test set.

## 7. MAE and RMSE

Errors use raw future voltage in mV. Patient-macro MAE gives each patient equal
weight regardless of their number of ECGs.

| ID | Architecture | Filter | MAE (mV) | RMSE (mV) | Patient-macro MAE (mV) | Runtime (s) |
|---|---|---|---:|---:|---:|---:|
| E1 | LSTM | P0 | 0.118466 | 0.251079 | **0.117035** | 62.1 |
| E2 | LSTM | P1 | 0.118542 | **0.251008** | 0.117076 | 50.0 |
| E3 | CNN-LSTM | P0 | 0.118491 | 0.251020 | 0.117066 | 38.0 |
| E4 | CNN-LSTM | P1 | 0.118515 | 0.251028 | 0.117081 | 38.6 |

Persistence and repeat-window had validation patient-macro MAE of 0.151984
and 0.173980 mV respectively. Runtimes are observed wall times, not a hardware
throughput benchmark. E1 ranked first on the stated metric; E2 had the lowest
pooled RMSE. Those metrics answer different questions.

## 8. LSTM preprocessing comparison

Filtering changed patient MAE by **+0.000042 mV** (95% paired interval
−0.000064 to +0.000133), a −0.035% reduction. The effect is inconclusive.

## 9. CNN-LSTM preprocessing comparison

Filtering changed patient MAE by **+0.000015 mV** (−0.000015 to +0.000044),
a −0.012% reduction. The effect is inconclusive.

## 10. LSTM versus CNN-LSTM

Without filtering, E3 minus E1 was **+0.000031 mV** (−0.000073 to +0.000120).
With filtering, E4 minus E2 was **+0.000004 mV** (−0.000063 to +0.000064).
The LSTM had lower point estimates in both comparisons; neither interval
supports an architecture advantage. All intervals resample validation patients
and do not include seed-to-seed variability.

## 11. Twelve-lead actual versus predicted ECG

E1 and E3 used the same example, validation ECG 30, selected as the lowest
common ECG ID. The local plot `week5_forecasts.png` shows the shared observed
history and true future. The Lead II zoom `week5_qrs.png` shows the first
detected eligible future QRS. Both plots are in the local pilot results folder
and illustrate one ECG.

## 12. Optional age/sex experiment

Adding age and sex to E1 changed patient-macro MAE from 0.117035 to 0.117056 mV
(−0.018% reduction); the paired change interval was −0.000081 to +0.000104 mV.
Pooled MAE/RMSE were 0.118479/0.251087 mV. There was no clear gain in this
four-epoch comparison. Week 4 feature-selection methods are separate.

## 13. Temporal occlusion

The local heatmap `week5_occlusion.png` uses E1
and one ECG from each of 12 validation patients. Masking aVF in the final
observed second gave the largest mean increase in error, about +0.000022 mV.
Masking V6 in the same interval reduced error by about 0.000031 mV. Both are
small perturbations compared with the unmasked 0.110568 mV MAE on these 12 ECGs.
The method describes sensitivity to the chosen mask, not clinical importance
or causality.

## 14. Limitations and conclusions

The study found no clear filtering or architecture advantage at four epochs.
The observed differences are small, estimated on one validation cohort with
one seed, and do not establish diagnostic fidelity. The real 120-epoch full
training runs and fold-10 evaluation remain pending. To show the recorded
primary runs locally in the notebook, set `study_id='week5_cpu_450x100_e4'`,
`epochs=4` and `smoke_test=False`; keep the remaining primary settings at
their defaults. To also show the completed smaller-cohort, shorter-budget and
age/sex results via their matching checks, enable the corresponding options
and set `shorter_epochs=2`.
