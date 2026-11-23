# Phase 1.5 acceptance criteria (fixed before running any check)

Written 2026-11-22 before the verification and download benchmarks were run.

## Data integrity: all must pass

| ID | Check | Pass if |
|---|---|---|
| C1 | Required files present | `ptbxl_database.csv`, `scp_statements.csv`, `RECORDS`, `LICENSE.txt`, `SHA256SUMS.txt`, `records100/` exist |
| C2 | Checksum list is authentic | `SHA256SUMS.txt` is byte-identical on two independent mirrors (PhysioNet web server and PhysioNet S3) |
| C3 | File count and checksums | every `records100/` and metadata file listed in `SHA256SUMS.txt` exists and matches its SHA-256 |
| C4 | No unexpected files | no files under `records100/` beyond the checksum list (e.g. no `.part` leftovers) |
| C5 | Record and patient counts | 21,799 ECG records and 18,869 patients, as published for v1.0.3 |
| C6 | Folder structure | each record has exactly one `.hea` and one `.dat` file at `filename_lr`, in folders of at most 1,000 records named by `ecg_id` |
| C7 | Headers | every header (all 10 folds) has 100 Hz, 12 signals, 1,000 samples, units mV, lead order I, II, III, aVR, aVL, aVF, V1–V6, and a record name matching its file |
| C8 | Signal format integrity | folds 1–9: every WFDB per-signal checksum matches the stored samples, and every physical signal is finite with shape 1000 × 12 |
| C9 | Metadata consistency | `ecg_id` unique; `strat_fold` in 1–10; `filename_lr` matches `ecg_id`; every SCP code is defined in `scp_statements.csv`; `RECORDS` agrees with the database; each patient's records sit in one fold |

Fold 10 is the sealed test set. It gets C3, C6 and C7 (checksums, file names and
headers), but its signal samples are not read (C8 excludes it).

## Download method comparison

All methods fetch the same file subset (`records100/00000/`: 987 records, 1,974 files,
28 MB) into an empty folder through the same network path. *(The first draft said
"1,000 records, 2,000 files"; that was an estimate, corrected when the file list was
built and before any benchmark ran.)* Methods that turn out to be sequential, with an
estimated run time over 5 minutes, run for 5 minutes, and their file rate is
extrapolated.

Measured for each method:
- wall time;
- number of files that pass SHA-256 verification afterwards;
- time for a second run (does it skip existing files?);
- whether an interrupted download resumes;
- whether admin rights or extra installs are needed.

The method recommended for an operating system must:
- produce 100% verified files;
- skip already-verified files on re-run;
- need no admin rights.

Among the methods meeting all three, the fastest is recommended.
