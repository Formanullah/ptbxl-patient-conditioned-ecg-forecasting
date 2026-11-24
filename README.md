# PTB-XL 12-lead ECG forecasting

Research-grade, leakage-free forecasting on PTB-XL v1.0.3: observe the first 7 s
of a native 100 Hz, 12-lead ECG and forecast the next 2.9 s. The final 100 ms of each
official record is a 500→100 Hz conversion artifact and is removed (decision D3.4).
Records with signal defects are excluded by fixed rules (D3.1). The project is built
phase by phase. Each phase ends with a report and a recorded go/no-go decision in the
[decision log](reports/DECISION_LOG.md).

**Consolidated report (Phases 0–19): [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md).**

**Study guide (reading order, glossary, self-test): [reports/STUDY_GUIDE.md](reports/STUDY_GUIDE.md).**

Presentation (32 slides with speaker notes): [reports/presentation/PTBXL_ECG_forecasting.pptx](reports/presentation/PTBXL_ECG_forecasting.pptx).

| Phase | Status | Report |
|---|---|---|
| 0 Forecasting feasibility | Done: 7→3 kept, with timing-aware evaluation (D0.8) | [report](reports/phase0_feasibility/REPORT.md) |
| 1 Reproducible foundation | Done: uv, typed YAML config, file-based tracking (D1.1–D1.8) | [report](reports/phase1_foundation/REPORT.md) |
| 1.5 Data acquisition and verification | Done: data complete and authentic; upstream `RECORDS` defect (D1.5.3) | [report](reports/phase1_5_data/REPORT.md) |
| 2 Dataset understanding | Done: fold 9 device-shifted; 8 independent leads; device drives difficulty | [report](reports/phase2_dataset/REPORT.md) |
| 3 Data quality | Done: 30 defect records excluded; last 10 samples trimmed (task now 7 s → 2.9 s); baseline wander drives difficulty | [report](reports/phase3_quality/REPORT.md) |
| 4 Leakage prevention | Done: official patient-level folds; labels and non-causal preprocessing leak the future (guards in code) | [report](reports/phase4_leakage/REPORT.md) |
| 6 Evaluation framework (done before 5, D5.0) | Done: MAE alone rewards flat lines; MAE + shape/morphology/rhythm gatekeepers | [report](reports/phase6_evaluation/REPORT.md) |
| 5 Preprocessing comparison | Done: raw mV, no augmentation; reference CNN beats flat but not `sn4`. Revised in Phase 7 (seeded re-run): input decomposition replicates a 0.34 µV gain and is adopted (D5.11) | [report](reports/phase5_preprocessing/REPORT.md) |
| 7 Data efficiency | Done: MAE gains diminish (floor ≈ 0.094 mV); more data does not fix beat timing; seeding bug found and fixed (D7.1) | [report](reports/phase7_data_efficiency/REPORT.md) |
| 8 Validation strategy | Done: select on fold 9 (ranking = 9-fold CV); old comparison rule had a 41% false-positive rate → 6 runs per arm + run-and-patient bootstrap (`compare_runs`, D8.3) | [report](reports/phase8_validation/REPORT.md) |
| 9 Model development | Done: CNN-LSTM best on MAE/DTW (−7.9% vs CNN, 9-fold CV) but worse beat morphology; no family beats `sn4`; CNN-LSTM + Transformer carried to Phase 10 | [report](reports/phase9_models/REPORT.md) |
| 10 Training objectives | Done: CNN-LSTM conditioned on the `sn4` forecast beats flat and L1 CNN-LSTM on MAE and every gatekeeper (9-fold CV, 9/9 folds); still worse than `sn4` on morphology and timing; spectral loss helps shape but costs MAE; derivative loss no effect | [report](reports/phase10_objectives/REPORT.md) |
| 11 Hyperparameters | Done: LSTM hidden 256 adopted (−0.77 µV MAE on fold 9, gatekeepers n.s.); spectral term trades MAE for morphology | [report](reports/phase11_tuning/REPORT.md) |
| 12 Robustness | Done: beats flat in every stratum and L1 in all but AF/flutter; CV shows hidden 256 costs template/timing vs 128 → final reverted to 128 (D12.1) | [report](reports/phase12_robustness/REPORT.md) |
| 13 Physiological plausibility | Done: correct beat count but damped (≈ 28% amplitude); limb-lead relations broken in 19% of records | [report](reports/phase13_plausibility/REPORT.md) |
| 14 Error analysis: patients | Done: AF/flutter, hypertrophy, pacing and baseline wander drive error; device effect vanishes after adjustment | [report](reports/phase14_patient_errors/REPORT.md) |
| 15 Error analysis: signals | Done: gain over L1 only for periodic histories; gate tracks periodicity | [report](reports/phase15_signal_errors/REPORT.md) |
| 16 Final model | Done: hidden-128 `sn4`-gated CNN-LSTM (D12.1), 6 seeds, frozen with hashes | [report](reports/phase16_final/REPORT.md) |
| 17 Test on fold 10 | Done (once): MAE 0.0850 mV; beats flat and L1 CNN-LSTM on MAE and every gatekeeper; still worse than `sn4` on morphology and timing | [report](reports/phase17_test/REPORT.md) |
| 18 Ablations | Done: skip path essential; gate trades shape for MAE; raw input beats the decomposition on fold 9 (kept, D18.1) | [report](reports/phase18_ablations/REPORT.md) |
| 19 Documentation | Done | [consolidated report](reports/FINAL_REPORT.md) |

## Setup (Windows, Linux, macOS; no admin rights needed)

1. Install [uv](https://docs.astral.sh/uv/):
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - Or, on any OS: `pip install --user uv`

   Behind a TLS-intercepting corporate proxy, also set `UV_SYSTEM_CERTS=1`. If large
   downloads time out, raise `UV_HTTP_TIMEOUT` (e.g. to 600 seconds).

2. Create the locked environment. uv installs Python 3.12 itself if it is missing.
   Choose exactly one torch build:

   ```sh
   uv sync --locked --extra cpu      # CPU laptops, macOS (Apple silicon: MPS)
   uv sync --locked --extra cu126    # NVIDIA, driver >= 525
   uv sync --locked --extra cu130    # NVIDIA, driver >= 580 (newest GPUs)
   ```

   Add `--extra tracking` for the optional MLflow UI. All builds use torch 2.14.1.

3. Download and verify the data: 43,604 files (100 Hz records + metadata, 544 MB),
   about 25 s here. Every file is checked against the official SHA-256 list; re-runs
   skip verified files, and interrupted downloads resume. The S3 mirror is used
   first, with physionet.org as fallback.

   ```sh
   uv run python scripts/download_ptbxl.py --root PTB-XL               # any OS
   sh scripts/download_ptbxl.sh --root PTB-XL                          # Linux/macOS, no project env needed
   powershell -ExecutionPolicy Bypass -File scripts\download_ptbxl.ps1 --root PTB-XL   # Windows
   uv run python scripts/download_ptbxl.py --root PTB-XL --verify-only
   uv run python research/phase1_5_verify_data.py                     # full integrity report
   ```

   Without Python, download the ZIP from the
   [PhysioNet page](https://physionet.org/content/ptb-xl/1.0.3/) in a browser and
   verify it later. Never build record lists from the dataset's `RECORDS` file; it is
   malformed (see the Phase 1.5 report).

4. Check the setup:

   ```sh
   uv run ptbxl-forecast env        # environment, git, lockfile, dataset fingerprint
   uv run pytest -q                 # 98 tests, ~50 s on CPU
   ```

## Configuration and runs

`configs/base.yaml` is validated against `src/ptbxl_forecast/config.py`. Unknown keys,
wrong types and overlapping train/val/test folds are rejected. Override values from
the command line:

```sh
uv run ptbxl-forecast config seed=123 runtime.threads=8
```

Code runs inside `ptbxl_forecast.tracking.Run`, which:
- seeds all random number generators and applies the runtime settings;
- writes `results/<study_id>/<run_name>_<config_hash>/` containing the resolved
  config, a provenance manifest (git commit, lockfile, packages, hardware, dataset
  fingerprint), per-step metrics, a summary and a log;
- refuses to overwrite a completed run.

`load_runs(results, study_id)` collects a study's runs into a DataFrame. Set
`tracking.mlflow=true` to mirror runs to MLflow, then view them with
`uv run mlflow ui --backend-store-uri sqlite:///results/mlflow.db`.

**Leakage rules** (Phase 4, enforced in `ptbxl_forecast.leakage` / `splits`): splits are
patient-level only (official folds); per-record preprocessing may use the history only;
fitted statistics come from training records only; PTB-XL statements, flags and reports
are never model inputs, because they describe the forecast window too.

Fold 10 is the sealed test set. `ptbxl_forecast.data.load_metadata` and `load_signals`
raise `SealedFoldError` for it unless `allow_test=True` (config: `data.allow_test`,
which is part of the config hash).

Bitwise reproducibility holds for the same hardware, torch build and
`runtime.threads`. Across machines, results are only statistically reproducible, so
reported results use multiple seeds.

## Reproducing the analyses

```sh
uv run python research/phase0_feasibility.py   # Phase 0, ~2 min on 38 cores; deterministic
uv run python research/phase0_figures.py
uv run python research/phase1_5_verify_data.py # Phase 1.5 integrity checks, ~15 s
uv run python research/phase2_dataset.py       # Phase 2, ~30 s (run Phase 0 first for device skill)
uv run python research/phase3_quality.py       # Phase 3, ~40 s (run Phase 0 first)
uv run python research/phase3_edge_artifact.py # needs: download_ptbxl.py --include records500/00000/ --include records500/01000/
uv run python research/phase3_cleaned_dataset.py # cleaned dataset statistics, trimming impact
uv run python research/phase4_leakage.py       # Phase 4, ~80 s
uv run python research/phase6_metrics.py       # Phase 6, ~90 s
uv run python research/phase5_signal_level.py  # Phase 5 (see its report for the full sequence)
```

## Layout

```
configs/            validated YAML configuration
src/ptbxl_forecast/ config, data access (sealed test fold, cleaning rules), quality rules, leakage guards, patient-level splits, metrics + patient-level statistics, preprocessing candidates, features, augmentations, ridge/CNN reference models, training loop, seeding, tracking, provenance, logging, CLI
tests/              unit tests
scripts/            data download/verification
research/           phase analyses and benchmarks
reports/            phase reports and the decision log
archive/            earlier implementations (Week 5 and before), unchanged
```

## Version control

- One branch per phase, merged after its go/no-go decision and tagged
  `phase-N-approved`.
- Committed: code, configs, `uv.lock`, reports.
- Never committed: data, caches, run folders, checkpoints.
- Reportable runs should set `tracking.require_clean_git=true`.
