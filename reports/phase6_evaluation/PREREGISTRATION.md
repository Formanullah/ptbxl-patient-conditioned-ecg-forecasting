# Phase 6 pre-registration: evaluation framework

Written 2026-11-22 before any Phase 6 computation. Data: cleaned folds 1–9 (19,571
records), task 7 s → 2.9 s (horizon 290 samples), raw-mV target (D3.1, D3.4, D4.3).
No model is trained.

## Candidate metrics (per record; aggregated patient-macro with 95% patient-cluster bootstrap CIs)

| Family | Metric | Definition |
|---|---|---|
| Pointwise | MSE, RMSE, MAE | over 290 × 12 samples, in mV² / mV |
| Correlation | Pearson r | per lead, averaged over leads; a constant forecast scores r = 0 |
| Energy | SNR (dB) | 10·log10(Σy² / Σ(y − ŷ)²) |
| Timing-tolerant | DTW-MAE | banded DTW (±100 ms) with \|·\| cost, divided by length; in mV |
| Morphology | Template MAE | median beat (−250 … +450 ms around R) of forecast vs truth, all leads; in mV. A forecast with no detectable beat gets a flat template |
| Morphology | R / ST / T amplitude error | from the templates: R peak; ST level at R + 80 ms; T-wave extreme at R + 150 … 400 ms; each relative to the PR level at R − 80 ms; in mV |
| Rhythm | Beat F1 | R peaks (lead II, detected on true history + forecast) matched to true R peaks within ±50 ms |
| Rhythm | RR error | \|mean RR predicted − mean RR true\| in ms (records with ≥ 2 beats in both) |
| Relative | Skill | 1 − metric(model) / metric(strongest baseline) |

## Probe forecasts (to test what each metric rewards)

- **Perturbations of the true target:**
  - time shift by 2, 5, 10 or 20 samples (20–200 ms);
  - amplitude ×0.8 and ×0.5;
  - low-pass at 15 Hz and at 5 Hz (morphology blurred);
  - level offset +0.1 mV;
  - white noise with SD 0.05 mV;
  - a beat removed (the target with one R-peak cycle replaced by the flat PR level).
- **Realistic forecasts:** history median (flat), `sn4` (repeat cycles), and the
  timing oracle (Phase 0 definitions, recomputed for the 290-sample horizon).

## Required properties (axioms)

| ID | Property | Applies to |
|---|---|---|
| A1 | A forecast with correct morphology and ≤ 50 ms timing error scores better than a flat line | metrics meant to reflect physiological plausibility |
| A2 | Monotone in shift size (0 < 20 < 50 < 100 < 200 ms) | all |
| A3 | Monotone in amplitude shrinkage (×1 < ×0.8 < ×0.5) | all except correlation-type metrics, which are scale-invariant by design (the property is reported) |
| A4 | Penalizes blurring (5 Hz low-pass worse than 15 Hz worse than none) | morphology metrics |
| A5 | Penalizes a missing beat | rhythm metrics |
| A6 | Defined for every record (no NaN), or with documented handling | all |
| A7 | Under 10 ms per record on one CPU core | all |

## Suite construction rule (fixed now)

1. A metric enters the suite only if it satisfies the axioms for its family.
2. Redundancy: a metric is dropped if \|Spearman ρ\| > 0.9 with an already-included
   metric across records, computed on the realistic forecasts. The more
   interpretable metric is kept.
3. Discriminative power is reported as the paired standardized difference between
   `sn4` and the median forecast.
4. The suite must cover four aspects: pointwise accuracy, timing-tolerant shape,
   morphology, and rhythm.
5. **Primary metric:** patient-macro MAE on the raw target, all 12 leads (continuity
   with Phase 0; the median is MAE-optimal for a constant). It stays primary **only
   if** a gatekeeping rule is added: a model counts as better than a baseline only if
   it also does not lose on the shape and rhythm metrics.
