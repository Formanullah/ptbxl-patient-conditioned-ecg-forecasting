# Phase 1 report: reproducible project foundation

Every tooling choice below was made by measurement on this machine (Linux x86-64,
40 cores, no GPU, corporate TLS-intercepting proxy), not by preference. The
benchmark scripts are in [research/phase1_benchmarks/](../../research/phase1_benchmarks/).

## 1. Verdict

| Requirement | Result | Evidence |
|---|---|---|
| One-command reproducible environment | ✅ `uv sync --locked --extra cpu` | 157 packages locked for all OSes; cold build 50 s, warm 0.9 s |
| Linux / Windows / macOS | ✅ Linux verified; ⚠️ Windows/macOS **configured but not yet executed** | Universal lock contains win/mac wheels; CI matrix runs all three on first push |
| CPU-only laptops and GPU machines | ✅ CPU verified; ⚠️ CUDA builds locked but unexecuted (no GPU here) | Extras `cpu`, `cu126`, `cu130` all pin torch 2.14.1 |
| Python versions | ✅ 3.11, 3.12 (pinned default), 3.14: 22/22 tests each | uv downloaded 3.11 itself, without admin rights |
| Fixed seeds, bitwise reproducibility | ✅ on same hardware/thread count; ❌ across thread counts (expected, §7) | Tests + `bench_determinism.py` |
| Configuration system | ✅ typed, strict, hashed | 8/8 invalid configs rejected in tests |
| Logging + experiment tracking | ✅ file store + optional MLflow mirror | Tests, incl. MLflow round trip |
| Earlier results still reproducible | ✅ Phase 0 re-run in the new locked environment | All 10 tables and `summary.json` **byte-identical** |

**Decision (2026-11-22): GO to Phase 1.5** (decision log D1.8). The Windows/macOS/GPU items stay open until CI
or a GPU run executes them (§9).

## 2. Constraints that shaped the choices

- **Corporate proxy with TLS interception.** Python tools that bundle their own CA list
  fail. uv needs `UV_SYSTEM_CERTS=1`, conda needs `ssl_verify: truststore`. pip and
  Poetry worked unchanged here.
- No admin rights assumed. No GPU. No conda. System Python is 3.9 (too old). `/tmp` is
  only 6 GB, and it filled up during the benchmarks.
- **PyTorch packaging.** The newest torch (2.15) ships only CUDA 13 GPU builds
  (driver ≥ 580). Its default Linux wheel from PyPI pulls in about 3 GB of CUDA
  libraries even on CPU-only laptops. torch 2.14.1 is the newest release with CPU,
  CUDA 12.6 and CUDA 13.0 wheels, so the same torch version runs on every machine type.

## 3. Environment management

Same six packages (numpy, scipy, pandas, matplotlib, wfdb, PyYAML), Python 3.12, cold
caches, times include resolution, download and install:

| Tool | Cold | Warm | Lockfile | Env size | Python provisioning | Notes |
|---|---:|---:|---|---:|---|---|
| **uv 0.13.7** | **2.7 s** | **0.2 s** | universal (all OS + extras), 1.2 s | 310 MB | yes, no admin | needs `UV_SYSTEM_CERTS=1` here |
| venv + pip | 35.5 s | 24.7 s | none (needs pip-tools) | 425 MB | no | |
| Poetry 2.2.1 | 26.9 s (lock 17.8 + install 9.1) | – | yes | – | no | needs a separately installed Python |
| conda 26.9.4 (Miniforge) | 62 s | – | needs conda-lock | 1.5 GB (+666 MB base) | yes | needs `ssl_verify: truststore` |

**Selected: uv.** It is 10–13× faster cold and >100× faster warm than the alternatives.
It is the only candidate that, in one tool and without admin rights, both locks all
platforms and installs the Python interpreter. It also supports PyTorch's per-variant
package indexes, via extras.

Rejected:
- conda: slowest, largest, and solves a non-Python dependency problem this project
  does not have.
- Poetry: 10× slower, and cannot provision Python.
- pip: no lockfile.

Risk: uv is younger than the others. Mitigation: `pyproject.toml` is standard
PEP 621 metadata, so pip or Poetry can still read it.

For the actual project (157 packages including CPU torch, MLflow, NeuroKit2, numba):
universal lock in 5.5 s; cold sync 50 s; warm rebuild 0.9 s.

## 4. Configuration management

| Candidate | Comments | Strict typed validation | CLI overrides | Composition | Footprint | Observed issues |
|---|---|---|---|---|---|---|
| Hydra 1.3.8 | ✅ | ✅ (structured configs) | ✅ | ✅ defaults lists, multirun | 5 pkgs, 6 MB, 155 ms import | creates `outputs/<date>/` per run; global app state; `@hydra.main` owns `main()` |
| **YAML + OmegaConf 2.3 schema** | ✅ | ✅ | ✅ dot-lists | ✅ ordered file merge | 3 pkgs (Hydra's own core) | none |
| YAML + PyYAML only | ✅ | ❌ needs custom code | ❌ | ❌ | 1 pkg | typos silently ignored |
| JSON (stdlib) | ❌ | ❌ | ❌ | ❌ | 0 | configs cannot carry rationale comments |
| TOML (stdlib `tomllib`) | ✅ | ❌ | ❌ | ❌ | 0 | read-only; no null; resolved config cannot be written back |

The OmegaConf schema, tested directly:
- It rejected a typo in a file key, a typo in a CLI key, a string given for an int,
  and a float given for an int.
- Each config build took 1.8 ms.
- The hash did not depend on key order.

**Selected: YAML files validated by a dataclass schema through OmegaConf** (`config.py`),
plus domain checks: train/val/test folds must not overlap, the window must fit in 10 s,
and the device must be known.

Hydra was rejected *for now*. Its distinctive features (multirun launchers, config
groups) are not needed before Phase 9/11. Seed-paired sweeps will be driven
explicitly from Python, and Hydra's per-run output folders would duplicate the
tracker's run folders. Upgrade path: the same YAML and dataclasses are valid Hydra
structured configs.

**Run identity.** `config_hash` is a SHA-256 over the *scientific* keys only. It
excludes `study_id`, `run_name`, `paths`, `logging` and `tracking`, so the same
experiment gets the same ID on every machine.

## 5. Experiment tracking

3 runs × 200 epochs × 10 metrics, then reading the runs back for analysis:

| Candidate | Install | Logging time | Local query | Local UI | Data leaves the machine? | Observed issues |
|---|---|---:|---|---|---|---|
| MLflow 3.18 (SQLite) | 89 pkgs, 627 MB | 3.4 s (≈ 0.56 ms/point) | ✅ 17–29 ms | ✅ | no | file store now **raises an error** ("maintenance mode"), so SQLite is required |
| mlflow-skinny 3.18 | 43 pkgs, 70 MB | – | ❌ | ❌ | no | cannot use SQLite, and the file store is disabled, so unusable locally |
| W&B 0.31.2 (offline) | 29 pkgs, 80 MB | 0.8 s | ❌ `UsageError` | ❌ | **yes**, to analyze | requires an account and upload to a third-party cloud |
| TensorBoard 2.21 | 13 pkgs, 127 MB (+ torch) | 0.4 s | curves only, 46 ms | ✅ | no | no run/parameter table; binary event files |
| JSON-lines files | 0 | 0.01 s | ✅ via pandas | ❌ | no | no UI |

**Selected: plain files as the source of truth, with an optional MLflow mirror**
(`tracking.py`, extra `tracking`).

- Each run folder holds `config.yaml`, `manifest.json`, `metrics.jsonl`,
  `summary.json`, `run.log` and `artifacts/`.
- `load_runs()` turns a study into one DataFrame for multi-seed statistics.
- With `tracking.mlflow=true`, the same parameters, metrics and tags go to a local
  SQLite MLflow database for browsing.

Rationale:
- The file store depends on nothing and will still be readable years from now. The
  MLflow file-store deprecation observed above shows that tool backends do change.
- MLflow adds 627 MB, which is reasonable as an opt-in but not as a requirement for
  CPU laptops.

W&B was rejected: offline runs cannot be analyzed without uploading project data to a
third-party service. TensorBoard was rejected because it cannot tabulate runs by
parameters, which multi-seed and ablation analysis needs.

## 6. Logging

The standard library's `logging` is used (`logs.py`). Each run gets a console handler
at the configured level, plus a DEBUG-level `run.log` in its run folder. Python
warnings are captured. loguru and rich were not adopted: they add dependencies and
offer nothing the record needs.

## 7. Seeds and determinism (measured)

Entering a `Run` seeds Python `random`, NumPy and torch from `cfg.seed`. It also
enables `torch.use_deterministic_algorithms`, disables cuDNN benchmarking, sets
`CUBLAS_WORKSPACE_CONFIG`, and applies the thread count *before* recording provenance.
DataLoader shuffling uses a dedicated seeded generator.

The checks used a CNN-LSTM sized like the real task (700×12 input, 300×12 output,
batch 64, hidden size 128):

| Check | Result |
|---|---|
| Same seed, same process type, `num_workers` 0 and 2 (fork, spawn, forkserver) | bitwise-identical weights |
| Same seed, two separate processes | identical weight hash |
| Thread count 1 / 4 / 8 / 16 | **four different weight hashes** |
| Deterministic mode overhead on CPU | 2050 vs 2054 ms/step (none measurable) |
| Need for a DataLoader `worker_init_fn` | none: torch 2.14 already seeds NumPy in workers (tested), so the helper was removed |

Consequences:
- **Bitwise reproducibility holds only for the same hardware, torch build and thread
  count**, because floating-point reductions are reordered otherwise. The thread count
  is therefore an explicit config value (`runtime.threads: 4`), is part of the config
  hash, and the value actually in effect is recorded in each manifest.
- Across machines (CPU vs GPU, different thread counts), results are only
  *statistically* reproducible. That is exactly why Phase 12 requires multiple seeds
  and confidence intervals.

## 8. Version control and data versioning

- **Code:** git. The repository root is the rebuilt project. The Week 5 code was moved
  with `git mv` to `archive/week5_4e9310a/` (761 files, history preserved).
- **Branches:** one branch per phase (`phase/N-name`), merged after the go/no-go
  decision, and tagged `phase-N-approved`.
- **Line endings:** `.gitattributes` forces LF, so files hash the same on every OS.
- **What is committed:** code, configs, `uv.lock`, and small reports, tables and
  figures. **Not committed:** dataset, caches, run folders, checkpoints.
- **Dataset:** PTB-XL v1.0.3 is an immutable, versioned release that publishes
  SHA-256 checksums, so DVC would add tooling without adding safety. Instead, each
  manifest records a fingerprint of the checksum entries for the files this study uses
  (43,600 files). `scripts/download_ptbxl.py --verify-only` re-verifies every file in
  9 s. Derived data, such as the patient splits from Phase 4, will be committed as
  small CSV files with their hashes.
- **Every run records:** git commit, dirty flag, hash of uncommitted changes, `uv.lock`
  SHA-256, package versions, OS/CPU/torch/CUDA details, config hash, and dataset
  fingerprint. `tracking.require_clean_git=true` refuses to start reportable runs from
  uncommitted code.

## 9. Cross-platform status

| Platform | Locked | Executed |
|---|---|---|
| Linux x86-64, CPU, Py 3.11 / 3.12 / 3.14 | ✅ | ✅ 22/22 tests each |
| Linux/Windows + CUDA 12.6 or 13.0 | ✅ | ❌ no GPU available |
| Windows x86-64, CPU | ✅ | ⚠️ in CI matrix, runs on first push |
| macOS arm64 (CPU / MPS) | ✅ | ⚠️ in CI matrix, runs on first push |
| macOS Intel | ❌ | – PyTorch no longer publishes macOS x86-64 wheels |

Portability measures already in place:
- `pathlib` everywhere, no shell scripts in the pipeline.
- The downloader uses only the Python standard library.
- UTF-8 file I/O is explicit.
- The determinism test covers the `spawn` start method that Windows and macOS use.

## 10. Project layout

```
pyproject.toml, uv.lock, .python-version   environment (D1.1)
configs/base.yaml                          validated configuration (D1.2)
src/ptbxl_forecast/
    config.py       schema, loading, overrides, config hash
    seeding.py      seeds, determinism, threads, device
    tracking.py     Run context manager, file store, MLflow mirror, load_runs
    provenance.py   git / environment / lockfile / dataset fingerprints
    logs.py         logging setup
    cli.py          `ptbxl-forecast env | config`
tests/              22 tests (config, determinism, tracking, provenance)
scripts/            data download and verification
research/           phase analyses (Phase 0) and benchmark scripts
reports/            phase reports + DECISION_LOG.md
archive/            earlier implementations, unchanged
```

## 11. Remaining risks

1. Windows/macOS/CUDA are locked but have not run yet. The first CI run and the first
   GPU run must be checked.
2. torch is pinned at 2.14.1 to keep CUDA 12 support. Moving to a newer torch is a
   deliberate decision, made with a re-lock and a recorded change of config hash.
3. GPU determinism has not been measured. Some CUDA kernels have no deterministic
   version; the code will raise rather than silently differ, and this will be checked
   in the first GPU run.
4. MLflow is optional and heavy. If it is not installed, `tracking.mlflow=true` fails
   with an installation hint.
