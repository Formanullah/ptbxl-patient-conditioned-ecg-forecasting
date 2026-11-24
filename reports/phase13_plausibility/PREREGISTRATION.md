# Phase 13 pre-registration: physiological plausibility

Written 2026-11-24 before any Phase 13 analysis. Data: the Phase 12 OOF forecasts
(final, L1), `sn4`, flat and the truth, folds 1–9.

## Measures (per record; summarised over records)

- **Lead consistency:** worst RMS violation of the Einthoven/Goldberger relations
  (Phase 5 `limb_residual`); median and share > 0.01 mV.
- **Amplitude:** lead-II peak-to-peak of the forecast / of the truth (median, IQR);
  R, ST and T amplitude errors of the median beat (Phase 6).
- **Timing bands (D6.6):** shares ≤ 25, 25–50, 50–150, > 150 ms; share with no detected
  beat; median ratio of detected to true beat count.
- **Spectral smoothness:** share of lead-II power above 15 Hz.
- **Range:** share of records with any sample > 0.5 mV outside the history's range.

## Hypotheses

| ID | Hypothesis |
|---|---|
| H13.1 | Final's share of records with limb violation > 0.01 mV ≤ L1's |
| H13.2 | Final's forecasts are damped: median lead-II peak-to-peak ratio < 0.8 |
| H13.3 | < 1% of final's forecasts leave the history range by > 0.5 mV |

Descriptive only otherwise; no model is changed in this phase.
