# Phase 3 pre-registration: data quality assessment

Written 2026-11-22 before the Phase 3 analysis ran. Scope: folds 1–9 (19,601
records), raw 100 Hz signals; fold 10 is sealed. This phase *measures* quality and
*recommends* what to do. Cleaning methods are compared in Phase 5.

Already known from earlier phases (disclosed):
- no NaN or invalid samples;
- no 16-bit ADC saturation;
- 18 records break the limb-lead relations;
- 6 records have a lead with SD < 0.02 mV;
- metadata missingness;
- human quality flags on 22.9% of records;
- device-dependent low-frequency content.

## Operational definitions (per record and lead)

| Issue | Index | Flag threshold |
|---|---|---|
| Signal dropout (missing data) | Longest run of identical consecutive samples | ≥ 50 samples (0.5 s) |
| Amplitude saturation / clipping | Longest plateau (identical samples) at the lead's own max or min, with \|value\| ≥ 1 mV | ≥ 5 samples (50 ms) |
| Implausible amplitude | max \|x\| | > 5 mV |
| Baseline wander | Peak-to-peak of the < 0.67 Hz component (zero-phase; for assessment only, never applied to model data) | > 1 mV |
| Baseline power | basSQI = 1 − P(0–1 Hz) / P(0–40 Hz) | < 0.95 |
| High-frequency noise | P(40–50 Hz) / P(0–50 Hz) | reported, no threshold |
| Kurtosis | kSQI (a clean ECG is peaky) | < 5 |
| QRS power | pSQI = P(5–15 Hz) / P(5–40 Hz) | outside 0.5–0.8 |
| Detector agreement (lead II) | bSQI: share of beats found by both NeuroKit2 and Elgendi (2010) within 150 ms | < 0.8 |
| Beat-shape consistency | tSQI: mean correlation of each beat with the median beat | < 0.5 |
| Burst noise | Largest 1 s RMS of the > 15 Hz component, divided by the record's median 1 s RMS | > 5 |
| Lead consistency | RMS residual of III = II − I, aVR = −(I + II)/2, aVL = I − II/2, aVF = II − I/2 | > 0.01 mV |
| Exact duplicates | SHA-256 of the integer ADC samples | any |
| Near-duplicates | Max normalized 12-lead cross-correlation over all lags | ≥ 0.99 |

Near-duplicates are checked for every pair of records from the same patient, and
across patients for candidate pairs pre-selected by a shift-invariant spectral
fingerprint (cosine ≥ 0.995).

The SQI thresholds follow the literature (Li et al. 2008; Clifford et al. 2012;
Behar et al. 2013). They were defined for higher sampling rates, so here they are
treated as screening thresholds, not ground truth.

**Record-level aggregation** (added before any computation):
- continuous per-lead indices use the median across the 12 leads;
- defect flags (dropout, plateau, amplitude, lead consistency) apply if any lead is
  affected;
- bSQI and the beat positions used for tSQI come from lead II.

## Hypotheses

**H3.1 — Signal indices agree with the human quality flags.** Each AUROC must be at
least the stated value:

| Index | Predicts flag | AUROC at least |
|---|---|---:|
| Baseline-wander peak-to-peak | `baseline_drift` | 0.75 |
| High-frequency noise index | `static_noise` | 0.70 |
| Burst index | `burst_noise` | 0.70 |

Evaluated on all records, and separately on human-validated records.

**H3.2 — Quality drives forecast difficulty.** Spearman ρ between baseline-wander
peak-to-peak and Phase 0's oracle-timing MAE is at least +0.3.

**H3.3 — No duplicates remain.** Zero exact-duplicate signals, and zero
near-duplicate pairs across different patients.

## Decision rule (fixed now, applied in Phase 5)

A quality issue leads to **excluding the record** only if both of these hold:

- it is a *data defect* that a deployed system would not normally receive (signal
  dropout, lead-relation violation, duplicate, implausible amplitude);
- it affects < 1% of records.

Noise, wander and other realistic artifacts are **kept**. They are handled by
preprocessing comparisons (Phase 5) and reported as strata in every evaluation.

Any exclusion is applied identically to all folds. Results are reported both with
and without the excluded records.
