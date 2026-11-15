# Data leakage controls and verification status

## Current verification status

**The software suite passed 60 tests after updating the configuration.** The new 4,500/600/100-patient cohort was prepared and its real-cache leakage audit passed all ten checks, including patient/record/window isolation, training-only statistics, future-mutation invariance and blocked fold-10 evaluation. It contains 5,193 / 677 / 114 recordings and 15,579 / 677 / 114 windows. The machine record is `outputs/custom_100hz_5s_3s_b088aaf9/leakage_audit.json`.

Both updated architectures passed one real-data training optimizer step and validation-batch inference with finite outputs/gradients. These checks verify execution, not forecast accuracy. Expanded-cohort neural training/evaluation remain pending. Test patients have not been evaluated. Existing repository-wide lint/formatting findings are separate from the passing software tests.

The previous 1,500/300/100 cohort also passed its audit and completed two 60-epoch runs plus validation/baseline evaluation. Its settings and measured report are retained separately; those scores are not evidence for the new cohort.

## Controls and tests to run

| Risk | Implemented control / intended verification |
| --- | --- |
| Patient overlap | Official folds and patient-set assertions; deliberate contamination test |
| Record/window overlap | Unique recording/window IDs, disjoint splits and bounded window starts |
| Leakage through scaling | Fit only training observed windows; compare refitted statistics |
| Leakage through imputation | Training-only age and expanded-feature estimates; explicit missingness |
| Future-derived static features | HR/RR use observed ECG only; future-mutation test |
| Lookahead filtering | Causal SOS filtering of observed windows only; no full-record forward/backward filter |
| Current/equal-time diagnoses | Assign history before incorporating equal-time labels; missing dates excluded |
| Feature-selection leakage | mRMR fits training features/targets; permutation ranks validation errors from a training-fitted model |
| Test-informed method selection | Checkpoint selection uses validation patient-macro MAE; fold-10 inference requires full mode and a frozen selection |
| Report mixing cohorts | Dataset-specific paths plus actual patient/window/source-metadata provenance checks |
| Stale checkpoint evaluation | Reporting excludes evaluation hashes that no longer match the selected checkpoint |
| Resume consistency | Save optimizer, scheduler, random-generator and DataLoader generator states |

Relevant passing tests are in `tests/test_pipeline.py` and `tests/test_reporting.py`, including changing-cohort, pending-state, stale-score and matched-comparison checks. Five warnings came from upstream WFDB/Pandas deprecated inplace operations.

## Historical evidence

Before cleanup, the earlier development implementation passed **52 tests** and real-cache leakage checks. Its complete raw-data audit checked **21,799 recordings** with zero invalid records. Machine records remain under `outputs/research_100hz_dev/`, including `leakage_audit.json` and `final_verification.json`.

Historical verification confirmed ten legacy checkpoint/result hashes unchanged and complete 60/100-epoch development runs. Those checks must not be relabeled as tests of the new report/refactor or of the larger training cohort.

## Corrections retained

The earlier cache guard caught an inherited configuration-path mismatch before overwrite. Named recipes apply after inheritance merging. Preprocessing now has one active implementation. Evaluation and report generation are separate from model training. Reports show completed measurements only for matching source metadata, actual IDs and window geometry.

Imputers/scalers/mRMR do not fit validation/test labels. Grouped permutation and hyperparameter selection explicitly use validation errors; they must not be described as training-only selection. Filtered observed inputs still predict unfiltered targets, with affine scaling reversed to recover mV.

## Limitations

Training windows overlap and are correlated; patient bootstrap is required for uncertainty. One seed and repeated validation comparisons can overfit the validation cohort. Detector peaks may be displaced and are not expert beat annotations. Historical diagnoses are a proxy for information available clinically. Previous examination of legacy test results means the overall project is not a historically pristine external test study, even though new method selection reserves fold 10. Clinical false-alarm reduction requires separate validation.
