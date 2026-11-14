# Change map

Starting commit: `2ad3e680fbe408345212f5d3b18bda2a115a1b45`.
Implementation branch: `work/ptbxl-7s3s-study`.

| Action | Decision |
| --- | --- |
| Keep | Official folds, WFDB validation, physical metrics, patient bootstrap, useful provenance and RNG restoration. |
| Simplify | Independent model classes, one training loop, one active configuration, shared CLI/Colab functions. |
| Fix | Consume the complete 10 seconds once, share the raw-input scaler, use early static conditioning, enforce a final selection lock, and exclude smoke scores from research tables. |
| Archive | Previous source, configurations, tests, documents and notebooks under `archive/legacy_2ad3e68`. Contrastive learning is historical rather than part of the new primary study. |

Existing results, caches, and weights were not moved. The archive's
`protected_artifacts.json` records SHA-256 hashes for 1,292 prior result/weight files.
Older 500-Hz, 5s-to-3s and context-length experiments remain distinct studies.
No previous checkpoint is converted to the new architecture or protocol.

The old active report references a 4,500-patient experiment and a 1,500-patient
report folder that are absent in this checkout. Its statements are preserved as
historical documentation, not verified results for the new study.

The baseline test suite passed 60 tests before the refactor. Current acceptance
checks are in `tests/`; the original tests are retained with the archived code.
ARIMA is explicitly deferred. The two feature-selection methods remain a separate,
validation-only expanded-feature extension.
