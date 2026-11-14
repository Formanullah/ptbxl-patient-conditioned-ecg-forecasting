# Data dictionary

| Field | Meaning and use |
| --- | --- |
| `ecg_id` | Unique recording ID; links waveform, metadata and result rows. |
| `patient_id` | Pseudonymous patient ID; determines group isolation and macro weighting. |
| `strat_fold` | Official fold; 1–8 train, 9 validation, 10 locked test. |
| `filename_lr` | Relative extensionless path beneath `records100/`; never `filename_hr`. |
| `recording_date` | Shifted timestamp; only strictly earlier records within the same patient may supply prior annotations. |
| `scp_codes` | Current-record annotations; excluded from that record's predictors. Optional strictly prior annotations are proxies. |
| Raw waveform | `[1000,12]` float32, physical mV, 100 Hz; verified with WFDB. |
| History / target | `[700,12]` / `[300,12]`; disjoint slices cover the complete record. |

Canonical lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
Header capitalization is normalized for comparison; unexpected order/units fail.

F0 contains only the twelve dynamic leads. F1 has four static values: training-
imputed/scaled age, age-missing flag, training-mode-imputed binary sex and sex-
missing flag. PTB-XL sex codes are 0=male and 1=female; other values are missing.
Age values outside 0–120, including anonymized values around 300, are missing.
Missingness flags remain unscaled. These fields condition a waveform forecast;
they do not establish a patient outcome prediction.

The separate expanded extension adds height (50–250 cm), weight (10–400 kg), BMI
(kg/m²), and observed-history HR (beats/min), RR mean (s), RR population std (s).
Unavailable/unreliable estimates become missing with explicit flags; continuous
variables use training median imputation and population-std scaling. XQRS-derived
HR/RR values are detector proxies, not expert measurements. Without optional
history this pool has 16 values; F0/F1 never require the detector.

Audited history optionally adds five prior diagnostic superclass indicators
(NORM, MI, STTC, CD, HYP) and a missing-history flag. Same-time and missing-date
records are excluded. F2 then has 10 static values; expanded has 22. These inputs
are disabled by default. Selection moves a variable with its flags as one group.
No feature-selection method removes any of the twelve primary ECG leads.

`cohort_manifest.csv` records exact ECG/patient/fold IDs, source filenames,
history/future lengths and whether waveforms were prepared. `run.json` records
feature order, fit statistics, protocol/source identity, active parameters and
runtime. A scaler snapshot and cohort snapshot accompany each checkpoint.
