# Phase 14 pre-registration: error analysis by patient factors

Written 2026-11-24 before any Phase 14 analysis. Data: Phase 12 OOF per-record scores.

- **Factors:** age band (<40, 40–59, 60–79, 80–89, >89), sex, device family, rhythm
  (Phase 12 categories), heart-rate band, diagnostic superclass (NORM, MI, STTC, CD,
  HYP; multi-label, each yes/no).
- **Per level:** final's patient-macro MAE and gatekeepers with patient-cluster
  bootstrap CIs; Δ MAE vs L1 (µV) and Δ timing vs `sn4` (ms), with CIs.
- **Adjusted analysis:** OLS of log per-record MAE on all factors jointly plus log
  baseline wander; patient-cluster bootstrap CIs (300 resamples); reference levels
  age 60–79, male, CS100, sinus, 60–100 bpm, superclass absent.
- **Question:** which factors explain error once device and baseline wander are held
  fixed (Phase 2 found the device drives difficulty)? Expected: device and baseline
  wander dominate; diagnoses have smaller adjusted effects. Exploratory; no decision
  depends on it.
