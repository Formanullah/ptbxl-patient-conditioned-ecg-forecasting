# Phase 1.5 report: data acquisition and verification

The criteria were fixed before any check ran; see [CRITERIA.md](CRITERIA.md). Raw
results are in [integrity.json](integrity.json) and
[table_download_methods.csv](table_download_methods.csv).

Reproduce:

```sh
uv run python scripts/download_ptbxl.py --root PTB-XL    # ~25 s on this network
uv run python research/phase1_5_verify_data.py           # ~15 s on 38 cores
```

## 1. Verdict

| ID | Check | Result | Evidence |
|---|---|---|---|
| C1 | Required files present | **PASS** | All present; the two changelogs were added to the download |
| C2 | Checksum list authentic | **PASS** | `SHA256SUMS.txt` is byte-identical locally, on physionet.org (no redirect) and on S3 (SHA-256 `b7224b92…`) |
| C3 | File count and checksums | **PASS** | 43,604 / 43,604 files match |
| C4 | No unexpected files | **PASS** | 43,598 files under `records100/`, all listed; no `.part` leftovers |
| C5 | Record and patient counts | **PASS** | 21,799 records, 18,869 patients |
| C6 | Folder structure | **PASS** | 22 folders, at most 1,000 records each; each record has one `.hea` and one `.dat`; no orphan files |
| C7 | Headers (all folds) | **PASS** | 21,799 / 21,799: 100 Hz, 12 leads, 1,000 samples, mV, standard lead order, format 16, gain 1000/mV |
| C8 | Signal integrity (folds 1–9) | **PASS** | 19,601 / 19,601 records: WFDB checksums match, shape 1000 × 12, all finite; ADC range −20,032 … 17,212, so no record reaches the 16-bit limits |
| C9 | Metadata consistency | **FAIL** (1 of 7 sub-checks) | The official `RECORDS` file is malformed (§3); the other 6 sub-checks pass |

**C9 failed because of a defect in the published `RECORDS` index, not in the data.**
Every signal and metadata file is complete, authentic and consistent. The study
indexes records through `ptbxl_database.csv`, which C5–C9 verified. The defect
therefore does not affect this project, but it is recorded as a failed
pre-registered check rather than redefined as a pass.

**Decision (2026-11-22): GO to Phase 2** (D1.5.5). Rule for all later phases: never build a record
list from `RECORDS` (D1.5.3).

## 2. What was verified beyond the published checksums

- **Authenticity of the checksum list (C2).** Every other check trusts
  `SHA256SUMS.txt`, so it was compared across two separately served copies:
  physionet.org (fetched without following redirects) and the S3 mirror. Both match
  the local copy.
- **Data version.** The v1.0.3 changelog lists 38 duplicate recordings removed from
  earlier versions. None of them is present. The remaining `ecg_id`s are exactly
  1 … 21,837 minus those 38, so there are no other gaps.
- **Format-level integrity (C8).** WFDB headers store a 16-bit checksum per lead.
  Recomputing it from the samples is independent of the SHA-256 list. Fold 10 got
  checksum, file-name and header checks only (C3, C6, C7); its samples were not read.
- **No patient spans two folds** (0 of 18,869). This is the property that Phase 4's
  leakage analysis will rely on and test in more depth.

## 3. Finding: malformed `RECORDS` index (upstream defect)

`RECORDS` should list 43,598 records (21,799 at 100 Hz and 21,799 at 500 Hz). It
contains 43,597 lines, because the newline between the last 100 Hz entry and the
first 500 Hz entry is missing:

```
records100/21000/21837_lrrecords500/00000/00001_hr
```

Splitting that line restores exact agreement with the database. Consequence for
anyone using it: **wfdb 5.0.0's `get_record_list('ptb-xl/1.0.3')` returns the fused
entry** (verified), so tools that iterate over it silently skip ECG 21837 (fold 9)
at 100 Hz and ECG 1 at 500 Hz. Not reported to PhysioNet from here: that is an external
action for the project owner, e.g. via PhysioNet's contact page, citing this section.

## 4. Download methods compared

Same 1,974 files (`records100/00000/`, 28 MB) per method, fresh folder, SHA-256
verification afterwards; Linux, through the corporate proxy. Methods that turned
out to be sequential ran for 300 s, and their rate was extrapolated.

| Method | Subset time | Full 100 Hz set | Skips existing | Resumes | Verifies | Needs |
|---|---:|---:|---|---|---|---|
| **Our Python script** (persistent connections, 32 threads) | **1.9–3.8 s** | **23 s (measured)** | yes, by hash | yes (hard kill tested) | **yes** | Python 3.9+ or uv |
| Our script, before the connection fix | 15.4 s | 8.9 min (measured) | yes | yes | yes | – |
| curl `--parallel` (32) | 0.9 s | ~20 s | **no** (re-downloads) | yes (`-C -`) | no | preinstalled on all three OS |
| wget URL list `-N` | 4.6 s | ~1.7 min | yes (timestamp) | yes | no | Linux only by default |
| AWS CLI v1 `s3 sync` | 7.1 s | ~2.6 min | yes (size/time) | yes | no | 52 s install; CA bundle behind proxy |
| PowerShell 7 `-Parallel` (32) | 45.1 s | ~17 min | file exists only | untested | no | PowerShell 7 (Windows ships 5.1) |
| PowerShell sequential | 4.4 files/s | ~2.8 h | no | no | no | – |
| **wget -r (PhysioNet's documented command)** | 1.1 files/s | **~10.9 h** | yes | yes | no | – |
| ZIP / browser (1.84 GB) | – | 44 s (35 s download + 9 s unzip) | n/a | yes (HTTP range) | after unzip | 1.3 GB of unused 500 Hz data |

**Selected for all three OS: `scripts/download_ptbxl.py`**, run directly or through
`download_ptbxl.sh` (Linux/macOS) or `download_ptbxl.ps1` (Windows). It is the only
method that meets all three criteria: verified files, skipping verified files on
re-run, and no admin rights.

Rejected:
- **curl** is faster but never verifies and re-downloads everything on re-run.
- **wget, AWS CLI and PowerShell** do not verify against the published checksums.
- **wget -r** from physionet.org would take about 11 hours.

**Fallback with no Python:** ZIP via a browser, then verify once Python or uv is
available.

### Mirrors

| Mirror | Observation | Role |
|---|---|---|
| PhysioNet S3 (`physionet-open`) | 0.22 s per request; fast under 32 parallel connections; no throttling seen | primary |
| physionet.org | 0.80 s per request. After the benchmark's load on it (a 32-thread run, then 5 min of `wget -r`), *all* requests timed out for several minutes; afterwards file URLs **302-redirected to S3** | fallback only |
| Google Cloud (`ptb-xl-1.0.3.physionet.org`) | 404 | not available |
| Kaggle and other re-uploads | need an account; unofficial copies | rejected: cannot be traced to the official release |

### Changes made to the downloader as a result

1. **Persistent connections** per thread, through the system proxy (environment
   variables or the Windows registry): **23× faster** for the full set (8.9 min → 23 s).
2. **Corrupted download → next mirror.** A wrong checksum is now retried on the next
   mirror. Before, the wrong file was kept.
3. **One failing file no longer aborts the run.** It is reported as `failed` with
   exit code 1.
4. **Redirects are followed.** This is needed since physionet.org started
   redirecting to S3.
5. `--include` (subset download) and `--mirror` (alternative mirror) options.
6. The changelogs are downloaded as metadata.

Each behavior has a test against a local HTTP server (6 tests in
`tests/test_download.py`). Three of them (mirror fallback, failure reporting,
connection reuse) fail on the original script. The other three guard behavior the
original already had or that urllib provided (skip on re-run, repair of a corrupted
local copy, redirects).

## 5. Facts passed on to later phases

| To | Fact |
|---|---|
| Phase 2/3 | Ages above 89 are stored as **300** (official v1.0.2 changelog) and must not be used as numeric age |
| Phase 3 | v1.0.3 removed 38 duplicates, some of whose pairs spanned folds (including the test fold). Phase 3 should still search for near-duplicate signals that remain |
| Phase 3 | No sample in folds 1–9 reaches the 16-bit ADC limits, so there is no digital clipping. Analog saturation still needs a separate check |
| Phase 4 | Patients map to exactly one `strat_fold` in the official split |
| All | Index records via `ptbxl_database.csv`, never `RECORDS` |

**Dataset record** (in `integrity.json`):
- PTB-XL v1.0.3, DOI 10.13026/kfzx-aw45;
- `SHA256SUMS.txt` SHA-256 `b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d`;
- 43,604 files and 544 MB used;
- records per fold 2,173–2,198; patients per fold 1,849–1,942.

## 6. Limitations

- Download timings come from one network, behind one corporate proxy, at one time
  of day, and only our script was timed more than once. Absolute times will differ
  elsewhere; large gaps (e.g. 23 s vs about 11 h) are unlikely to reverse.
- The Windows wrapper was tested with PowerShell 7 on Linux, not with Windows
  PowerShell 5.1 on Windows. The script uses only commands that exist in 5.1.
- The physionet.org outage and redirect were observed, not explained. They may
  reflect rate limiting triggered by the benchmark itself.
