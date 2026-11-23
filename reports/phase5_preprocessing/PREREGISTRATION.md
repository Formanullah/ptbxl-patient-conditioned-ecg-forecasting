# Phase 5 pre-registration: preprocessing comparison

Written 2026-11-22 before any Phase 5 computation. Training: cleaned folds 1–8
(17,392 records). Comparison: fold 9 (2,179 records). Fold 10 sealed. Scoring: the
Phase 6 suite and rule (D6.2): MAE primary; DTW-MAE, template MAE and beat timing as
gatekeepers. Every per-record transform sees the history only, and every fitted
statistic comes from training records only (D4.3, D4.4). The target stays raw mV;
target scaling is inverted before scoring.

## Reference learners (D5.1)

- **Ridge:** direct multi-output linear forecaster, history (700 × 12, after the
  candidate preprocessing) → 290 × 12 target.
  - The ridge λ is chosen on a 20% patient-level holdout of the training folds, from
    a fixed grid; then the model is refit on all training records.
  - Fold 9 is used only to compare candidates.
- **CNN:** small 1-D CNN (3 strided conv blocks, width 64, about 1.8 M parameters),
  L1 loss, AdamW at learning rate 1e-3, batch 64, up to 40 epochs.
  - Early stopping (patience 5) on the same internal patient holdout.
  - Seeds 42, 123, 999. Run for the shortlist and for augmentation.

## Candidates

| Category | Candidates |
|---|---|
| Missing values | not applicable (D5.2) |
| Noise / baseline | none; causal Butterworth high-pass 0.5 Hz; causal Butterworth band-pass 0.5–40 Hz; zero-phase band-pass 0.5–40 Hz on the history only; causal FIR low-pass 40 Hz (linear phase); wavelet denoising (db4, soft universal threshold, history only); median-filter baseline removal (200 + 600 ms, history only); baseline/ECG decomposition (causal 0.67 Hz low-pass component + residual, both as inputs) |
| Scaling | none (mV); global standard; lead-wise standard; lead-wise robust (median/IQR); lead-wise min-max; per-record history z-score (inverted on the output) |
| Features (ridge only) | raw; + statistical (per-lead mean, SD, min, max, skewness, kurtosis); + morphological (mean/last/SD of RR, time since last R peak, median beat template; history-only R peaks); + frequency (Welch PSD bins); hybrid (all) |
| Dimensionality | 12 leads; 8 independent leads in and out (III/aVR/aVL/aVF derived exactly); PCA on the ridge inputs (95%, 99% variance); nonlinear autoencoder, UMAP: see the decision rule; t-SNE for analysis only |
| Augmentation (CNN only) | none; Gaussian noise (input, σ = 0.02 mV); magnitude scaling (×U[0.8, 1.2], input and target together); baseline-wander injection (0.05–0.5 Hz, up to 0.3 mV, input and target); time warping (±5% resample of the whole record); masking (two random 100 ms input segments); mixup (α = 0.2); lead dropout (one input lead zeroed) |

## Signal-level criteria (filters; no learner)

**S1 — IEC 60601-2-25 impulse test.** A 3 mV × 100 ms pulse followed by flat signal
must give a baseline displacement ≤ 0.1 mV and a slope ≤ 0.3 mV/s outside the pulse.
Applied as the filter would be used: a causal filter is evaluated after the pulse; a
history-only zero-phase filter is evaluated on a history whose pulse sits mid-window.

**S2 — Distortion and morphology** on 500 low-noise training records (kSQI ≥ 15,
baseline wander ≤ 0.1 mV) with synthetic noise added:
- baseline wander: 0.1–0.4 Hz, 0.5 mV;
- EMG-like noise: 20–45 Hz band-limited, 0.05 mV RMS.

Measured on the filtered history, against the clean history:
- RMSE;
- median-beat template correlation;
- ST-level error at R + 80 ms;
- runtime per record.

A filter is **morphology-preserving** if the median template correlation is ≥ 0.99 and
the median ST error is ≤ 0.05 mV.

## Decision rule (fixed now)

1. **Default** is the simplest pipeline: raw mV, no filter, no scaling, 12 leads, no
   features, no augmentation.
2. **Adopting a candidate requires both:**
   - it beats the default on fold 9 under the D6.2 rule (paired patient-cluster
     bootstrap) with ridge; and, for sequence transforms in the shortlist (up to the
     two best filters, the best scaling, the 8-lead variant),
   - it also beats the default with the CNN (seed-averaged predictions, same rule).
3. **Filters must also pass S1 and S2.** A filter failing them is rejected even if it
   lowers the error, because it changes ECG morphology.
4. **Augmentations** are judged with the CNN only. An augmentation is adopted if it
   beats "none" under D6.2.
5. **Nonlinear autoencoders and UMAP** are only built if PCA shows a benefit; a
   linear autoencoder is equivalent to PCA. t-SNE is used only to visualize history
   representations, coloured by device and rhythm.
6. Each adopted step is reported with its effect size and CI. Interactions between
   adopted steps are checked once, with the CNN, as a combined pipeline.
