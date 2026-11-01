# Configuration, scripts, and execution guide

This guide explains how to operate the ECG forecasting project. All commands use
Windows PowerShell from the project root. Start with the small pilot; use a new
configuration and new output directories for each changed experiment.

## 1. What the project does

At 500 Hz, the model reads 300 samples (0.6 seconds) from each of 12 ECG leads and
predicts the following 150 samples (0.3 seconds). Optional static inputs describe
age, sex, and diagnoses from strictly earlier recordings of the same patient.

The complete pipeline is:

```text
PTB-XL waveforms + metadata
  -> audit raw data
  -> preprocess and normalize using training input windows
  -> create patient-isolated training/validation/test loaders
  -> train LSTM and CNN-LSTM variants
  -> select checkpoints by validation forecast MSE
  -> evaluate the selected models on test data
  -> compare with baselines and produce figures and documents
```

## 2. Files and scripts: what each one does

Use `python -m tsa <command>` to operate the package. The files inside `src/tsa/`
are reusable modules; do not launch them directly as standalone scripts.

| File or folder | Purpose | How you use it |
| --- | --- | --- |
| `src/tsa/__main__.py` | CLI command routing and writer lock | `python -m tsa --help` |
| `src/tsa/config.py` | Loads YAML, validates settings, defines leads/features/variants and protocol hashes | Imported by the pipeline |
| `src/tsa/data.py` | Audits WFDB signals, splits patients, builds historical features, normalizes, caches arrays, creates DataLoaders | `audit`, `preprocess` |
| `src/tsa/models.py` | LSTM, CNN-LSTM, static fusion, ECG augmentations and NT-Xent loss | Used by training |
| `src/tsa/training.py` | Optimization, checkpoint selection/resume, inference, smoke runs, baselines and timing | `train`, `evaluate`, `smoke`, `baselines`, `benchmark`, `run-experiments` |
| `src/tsa/evaluation.py` | Voltage metrics, subgroup/lead/horizon plots, comparisons and bootstrap intervals | Used by training; `compare` |
| `src/tsa/reporting.py` | Assignment report, presentation notes, metadata figure and Colab ZIP | `report`, `bundle` |
| `src/tsa/__init__.py` | Package metadata | No direct execution needed |
| `tools/status.py` | Prints run progress and writes `TRAINING_STATUS.md` | `python tools/status.py --config configs/pilot.yaml` |
| `tools/make_notebooks.py` | Rebuilds both notebooks, replacing their existing contents | `python tools/make_notebooks.py`; only needed when editing notebook generation |
| `tests/test_pipeline.py` | Automated integrity, model, loss, metric and resume checks | `python -m pytest -q` |
| `configs/pilot.yaml` | Small experiment; default CLI configuration | Pass with `--config`, or omit the flag |
| `configs/base.yaml` | Full-data experiment | Pass explicitly with `--config configs/base.yaml` |
| `notebooks/assignment_walkthrough.ipynb` | Interactive inspection of arrays, tensor shapes and results | Open in Jupyter/VS Code |
| `notebooks/colab_training.ipynb` | GPU training using the same package and a prepared cache | Open in Google Colab |
| `pyproject.toml` | Package dependencies, test and lint settings | Installed through `requirements.txt` |
| `requirements.txt` | Installs the local package with development dependencies | `pip install -r requirements.txt` |
| `requirements-tested.txt` | Records versions tested locally | Use as a version reference |
| `.github/workflows/ci.yml` | Runs tests, lint and formatting checks on GitHub pushes/PRs | Automatic; does not train models |

## 3. Setup

```powershell
cd D:\Study\HIS_Project\Project\ptbxl_assignment
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m tsa --help
```

The commands call the environment's Python directly, so PowerShell activation is
unnecessary. On Linux/macOS, replace `.\.venv\Scripts\python` with `.venv/bin/python`.
On Colab, use its GPU PyTorch rather than installing the CPU wheel above.

The raw dataset directory should contain:

```text
PTB-XL/
  ptbxl_database.csv
  scp_statements.csv
  records500/...
  records100/...    # only required if using 100 Hz
```

The GitHub repository contains code, documents and pilot CSV/PNG results. Raw data,
preprocessing arrays, trained checkpoints and predictions must be prepared locally
or obtained through your own saved artifacts. The paper PDF stays local.

## 4. First check: automated tests

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests tools
.\.venv\Scripts\python -m ruff format --check src tests tools
```

The current suite has 18 test cases. It creates synthetic ECG fixtures and does not
need the full PTB-XL dataset. It tests patient isolation, historical chronology,
missingness, train-only statistics, future-sample isolation, inverse normalization,
model dimensions/gradients, contrastive masks, metrics, cache tampering, reproducible
patient subsets and checkpoint resume.

If pytest cannot access the Windows temporary directory, use a fresh local directory:

```powershell
.\.venv\Scripts\python -m pytest -q --basetemp .pytest-temp/run1 -p no:cacheprovider
```

Use this directory only for pytest temporary files; pytest may clear its contents.

## 5. Prepare real data

From a fresh clone with raw PTB-XL available, run these in order:

```powershell
.\.venv\Scripts\python -u -m tsa audit --config configs/base.yaml
.\.venv\Scripts\python -u -m tsa preprocess --config configs/base.yaml
.\.venv\Scripts\python -m tsa preprocess --config configs/pilot.yaml
```

`audit` reads the raw recordings and checks sampling rate, dimensions, lead order,
units, finite values and patient/fold metadata. It writes an explicit failure list
and stops if invalid records exist. Even with a pilot config, audit checks all raw
records; it does not restrict itself to the pilot cohort.

Full preprocessing extracts the fixed windows, builds static features and saves the
full cache. Pilot preprocessing selects whole patients from that full cache and
refits normalization on the selected training cohort. It also needs the raw metadata
CSVs. The source cache must match the requested sampling rate and window geometry.

If valid caches already exist locally, skip rebuilding them. Training validates
their fingerprints and file hashes before using them.

The official folds are 1–8 for training, 9 for validation and 10 for testing. Patients
cannot cross split boundaries. The pilot selects 450/100/100 patients, producing
515/108/116 recordings for the current seed and dataset.

## 6. Run a short real-data check

```powershell
.\.venv\Scripts\python -m tsa smoke --config configs/pilot.yaml --model lstm --variant patient_history
```

This uses at most 128 recordings per split and at most two epochs. It checks the
complete training/evaluation path and saves results beneath separate `smoke`
directories. Smoke results are debugging outputs, not assignment comparison results.

Optionally estimate training time:

```powershell
.\.venv\Scripts\python -m tsa benchmark --config configs/pilot.yaml
```

The benchmark measures one short training pass for each architecture with and without
contrastive loss. It writes `benchmark.json` under the experiments directory. Estimates
exclude validation, checkpointing and plotting; this command does not evaluate test
performance or save model weights.

## 7. Train, compare and report

Run all eight models, both baselines and the report:

```powershell
.\.venv\Scripts\python -u -m tsa run-experiments --config configs/pilot.yaml
```

Or train one model:

```powershell
.\.venv\Scripts\python -u -m tsa train --config configs/pilot.yaml --model lstm --variant patient_history
.\.venv\Scripts\python -u -m tsa train --config configs/pilot.yaml --model cnn_lstm --variant patient_history_contrastive
```

| `--model` | Meaning |
| --- | --- |
| `lstm` | Two-layer recurrent encoder |
| `cnn_lstm` | Temporal convolutions, pooling and the recurrent encoder |

| `--variant` | Static vector | Auxiliary contrastive loss |
| --- | ---: | --- |
| `ecg_only` | 0 fields | No |
| `demographics` | 4 fields | No |
| `patient_history` | 10 fields | No |
| `patient_history_contrastive` | 10 fields | Yes |

Training evaluates the selected model and generates its plots automatically. For
individual follow-up operations:

```powershell
.\.venv\Scripts\python -m tsa baselines --config configs/pilot.yaml
.\.venv\Scripts\python tools/status.py --config configs/pilot.yaml
.\.venv\Scripts\python -m tsa evaluate --config configs/pilot.yaml --model lstm --variant patient_history
.\.venv\Scripts\python -m tsa compare --config configs/pilot.yaml
.\.venv\Scripts\python -m tsa report --config configs/pilot.yaml
```

`baselines` evaluates final-value persistence and repetition of the last observed
window. `status.py` writes the progress document as well as printing progress.
`evaluate` uses an existing best checkpoint; if saved test metrics already exist,
it returns those metrics rather than rerunning inference. `compare` reads existing
run artifacts and writes comparison tables/charts and paired bootstrap intervals.
`report` uses available results to write the report and presentation notes; it does
not train missing runs or recreate all per-model plots. It also refreshes the root
`PRESENTATION_NOTES.md`. A complete comparison needs all eight neural runs and both
baselines.

## 8. Configuration reference

Edit YAML using spaces, not tabs. Paths are interpreted from the current working
directory, so run commands from the project root. The only CLI overrides are
`--root` and `--device`; edit the YAML for other settings.

| Setting | Pilot default | Meaning and change implications |
| --- | --- | --- |
| `profile` | `pilot` | Human-readable run label |
| `data.root` | `PTB-XL` | Raw dataset location |
| `data.cache` | `data/pilot` | Prepared arrays and manifest |
| `data.sampling_rate` | `500` | 500 or 100 Hz; assignment protocol uses 500 |
| `data.history` | `300` | Observed samples; 0.6 seconds at 500 Hz |
| `data.horizon` | `150` | Future samples; 0.3 seconds at 500 Hz |
| `data.start` | `0` | First observed sample |
| `subset.source_cache` | `data/preprocessed` | Matching full cache used to select patients |
| `subset.seed` | `42` | Patient-selection randomness |
| `subset.patients.train/val/test` | `450/100/100` | Patient counts, not recording counts |
| `model.hidden` | `128` | LSTM hidden-state size |
| `model.depth` | `2` | Number of LSTM layers |
| `model.dropout` | `0.2` | Between recurrent layers and in forecast head; recurrent dropout is inactive with one layer |
| `model.static_embedding` | `32` | Width of static-feature projection |
| `training.seed` | `42` | Initialization, shuffling and augmentation seed |
| `training.epochs` | `5` | Maximum epochs; full configuration uses 25 |
| `training.patience` | `3` | Stop after this many epochs without validation improvement; full uses 5 |
| `training.batch_size` | `64` | Recordings per batch |
| `training.lr` | `0.001` | AdamW learning rate |
| `training.weight_decay` | `0.0001` | AdamW regularization |
| `training.gradient_clip` | `1.0` | Maximum gradient norm |
| `training.num_workers` | `0` | Loader processes; zero is reliable on Windows |
| `training.threads` | `4` | PyTorch CPU threads |
| `training.device` | `auto` | CUDA if available, otherwise CPU; also accepts `cpu` or `cuda` |
| `contrastive.weight` | `0.1` | Weight in forecast MSE + weight × NT-Xent |
| `contrastive.temperature` | `0.1` | Similarity scaling in NT-Xent |
| `contrastive.jitter_std` | `0.01` | Noise standard deviation in normalized voltage units |
| `contrastive.gain_min/gain_max` | `0.95/1.05` | Amplitude gain range shared across leads within each view |
| `evaluation.bootstrap_samples` | `1000` | Patient resamples for paired confidence intervals |
| `paths.experiments` | `experiments/pilot` | Model states, predictions and logs |
| `paths.outputs` | `outputs/pilot` | Figures and CSV comparisons |
| `paths.documents` | `Project_Documents/pilot` | Report, presentation notes and status |

The observation and forecast windows must fit inside the ten-second record:
`start + history + horizon <= 10 * sampling_rate`. Changing sampling rate, window
geometry or patient selection requires a new compatible cache. Changes to CNN
channels/kernels or the 256-unit forecast head require editing `models.py`; these
are not YAML options.

One-command overrides, which do not edit YAML:

```powershell
.\.venv\Scripts\python -m tsa audit --config configs/base.yaml --root 'D:\Datasets\PTB-XL'
.\.venv\Scripts\python -m tsa train --config configs/pilot.yaml --model lstm --variant patient_history --device cpu
```

Set the root in every relevant YAML if the location should persist across commands.
Use `--device cuda` only when CUDA-enabled PyTorch and a compatible GPU are available.

## 9. Increase the experiment safely

Create a copy rather than changing the completed pilot:

```powershell
Copy-Item configs\pilot.yaml configs\medium.yaml
```

In `configs/medium.yaml`, change these fields; keep the other copied fields:

```yaml
profile: medium
data:
  cache: data/medium
subset:
  seed: 42
  patients:
    train: 1500
    val: 300
    test: 100
training:
  epochs: 15
  patience: 5
paths:
  experiments: experiments/medium
  outputs: outputs/medium
  documents: Project_Documents/medium
```

This is an edit checklist, not a complete YAML file: retain `root`, window settings,
`source_cache`, model settings and all remaining fields from the copied pilot.
Keep the test patient count and subset seed fixed during development. Increasing
counts with the same seed keeps earlier selected patients. Use validation results
to choose settings; test results should serve as final comparisons.

```powershell
.\.venv\Scripts\python -m tsa preprocess --config configs/medium.yaml
.\.venv\Scripts\python -m tsa smoke --config configs/medium.yaml
.\.venv\Scripts\python -u -m tsa run-experiments --config configs/medium.yaml
.\.venv\Scripts\python tools/status.py --config configs/medium.yaml
```

For full data, use `configs/base.yaml`; it has no `subset` section. Existing local
full-data checkpoints are preserved, so running that profile can resume earlier
work. Use new experiment/output/document paths when you want a fresh full-data run.

## 10. Resume and configuration changes

Press Ctrl+C to stop, then rerun the identical training command. Resume starts from
the last completed epoch. Work partway through an interrupted epoch is repeated.
`last.pt` includes weights, optimizer state, random-generator states and history.
`best.pt` contains the validation-selected weights. Completed runs are skipped.

Scientific settings are part of the checkpoint signature, including epoch count,
learning rate and seeds. Increasing `epochs` in an already used configuration is
not supported as a same-directory resume. Use a new experiment directory and train
fresh for a changed configuration. Device, thread count, worker count and storage
paths are excluded from the scientific signature for portability; numerical results
can still vary across hardware/software.

Only one preparation/training writer is allowed at a time. If the pipeline reports
another active job, inspect status and the running process before starting another.
An idle `.pipeline.lock` file can remain after completion; its existence alone does
not mean a job is running.

## 11. Where results are saved

For `lstm` + `patient_history` with the pilot configuration:

```text
data/pilot/
  X_*.npy, y_*.npy, static_*.npy
  metadata_*.csv, manifest.json

experiments/pilot/lstm_patient_history/
  config.json                 Settings, versions, parameter counts and device
  feature_manifest.json       Features and preprocessing provenance
  last.pt                     Resumable state
  best.pt                     Selected weights
  history.csv                 Epoch losses and durations
  test_metrics.json           Overall metrics and training summary
  predictions.npz             Forecasts, targets and record/patient IDs
  per_record_metrics.csv      Record-level errors and subgroup metadata

outputs/pilot/
  model_comparison.csv, model_comparison.png
  paired_patient_bootstrap.csv
  metadata_distributions.png
  lstm_patient_history/
    learning_curves.png
    per_lead_metrics.csv, per_lead_errors.png
    subgroup_metrics.csv, subgroup_errors.png
    horizon_metrics.csv, horizon_error.png
    forecast_lowest_id.png
    forecast_median_error.png
    forecast_worst_error.png

Project_Documents/pilot/
  ASSIGNMENT_REPORT.md
  PRESENTATION_NOTES.md
  TRAINING_STATUS.md          Written by tools/status.py
```

MAE and RMSE are in mV; MSE is in mV squared. Validation/training loss curves use
normalized units, so their numerical values are not directly comparable with test
voltage MSE. Lower error is better. R² can be negative. Patient-macro MAE gives each
patient equal weight. Bootstrap differences are B minus A; negative favors B.

The current pilot has all eight neural runs and two baselines, with five epochs
per neural run. Persistence has lower MAE than the neural models in this pilot.
Treat these as preliminary findings; the short observed window, sparse prior
history, small cohort and one seed limit conclusions.

## 12. Colab workflow

Build the transferable cache/code bundle locally:

```powershell
.\.venv\Scripts\python -m tsa bundle --config configs/pilot.yaml
```

Upload `outputs/pilot/colab_bundle.zip` manually to
`MyDrive/ptbxl_assignment/`. Open `notebooks/colab_training.ipynb`, select a GPU
runtime and run the cells in order. The notebook derives its configuration from
the pilot and checks GPU availability. Arrays run from Colab local disk; training
artifacts persist under `MyDrive/ptbxl_assignment/pilot/`.

The bundle includes the selected processed cache and code. It excludes the PDF,
raw data and model checkpoints, and removes the private manuscript comparison from
the bundled reporting module. No upload occurs when the bundle command runs.
Keep CPU and GPU run artifacts in separate experiment directories when comparing
fresh runs, then generate the report using the matching configuration.

## 13. Common problems

| Message or symptom | Action |
| --- | --- |
| `No module named tsa` | Use `.venv` Python and install `-r requirements.txt` from the root |
| Metadata or WFDB file not found | Check `data.root`, folder layout and `filename_hr` paths |
| Cache manifest not found | Prepare the matching full cache, then the pilot cache |
| Cache protocol mismatch | Use a matching config/cache; rebuild under a new cache path for changed geometry/cohort |
| Cached file or metadata changed | Verify the intended source and rerun preprocessing; use fresh run directories for the rebuilt cache |
| Configuration changed for run | Restore the original YAML to resume, or use a new experiments directory for fresh training |
| `SKIP completed` | Saved results already exist; use a new experiment path for a fresh run |
| No selected checkpoint | Train that model/variant first; a GitHub clone does not include weights |
| CUDA unavailable or out of memory | Select CPU, or use CUDA-enabled PyTorch/compatible GPU; reduce batch size in a fresh experiment |
| Another job active | Inspect existing work and wait for it to finish; do not launch duplicate writers |
| Missing report figures after cloning | Pilot CSV/PNG figures are tracked; new-profile figures require evaluating its models locally |

## 14. Short command checklist

For a fresh setup: install -> tests -> audit full raw data -> preprocess full ->
preprocess pilot -> smoke -> run-experiments -> inspect report/status.

With valid local caches and completed pilot results, start with:

```powershell
.\.venv\Scripts\python tools/status.py --config configs/pilot.yaml
.\.venv\Scripts\python -m tsa compare --config configs/pilot.yaml
```

Read the scientific findings in [the pilot assignment report](pilot/ASSIGNMENT_REPORT.md)
and use [the presentation notes](pilot/PRESENTATION_NOTES.md) to explain CNNs, LSTM
gates, training and the measured comparisons in class.
