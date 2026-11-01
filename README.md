# Contrastive Patient-Stratified ECG Forecasts

Predict the next **150 samples (0.3 seconds)** of all 12 PTB-XL ECG leads from
**300 observed samples (0.6 seconds)** at 500 Hz. Compare LSTM and CNN-LSTM with
age, sex, strictly earlier diagnostic superclasses, and optional contrastive learning.

## Start here

All current project documents are together in **`Project_Documents/`**.
Choose one based on what you need; you do not need to read duplicate versions.

| Order | Document | Use it for |
| --- | --- | --- |
| 1 | [Configuration and Run Guide](Project_Documents/CONFIGURATION_AND_RUN_GUIDE.md) | Run locally step by step, configure data/models, change parameters, and compare experiments |
| 2 | [Assignment Report](Project_Documents/ASSIGNMENT_REPORT.md) | Understand preprocessing, methods, measured results, figures, and literature comparison |
| 3 | [Presentation Notes](Project_Documents/PRESENTATION_NOTES.md) | Explain CNN operations, LSTM gates, tensor shapes, training and findings in class |

**The report and notes describe the completed small-data experiment.**
“Pilot” means that smaller preliminary run, created to reduce CPU training time.
It uses 450 training, 100 validation and 100 test patients, with at most five epochs.
Full-data training uses the same pipeline with `configs/base.yaml` and up to 25 epochs.
The guide explains how to choose either configuration.

## Current results

All eight small-data neural runs and both baselines are complete.
The best neural MAE was **0.150113 mV** (LSTM with patient history), while persistence
achieved **0.141546 mV**. These preliminary results do not establish a benefit from
conditioning or contrastive learning. The short input window and single seed limit
conclusions; clinical false-alarm reduction remains a motivation requiring validation.

Open the [comparison table](outputs/pilot/model_comparison.csv) or
[comparison chart](outputs/pilot/model_comparison.png) for the measured results.

## Project folders

```text
src/tsa/             Reusable data, models, training, evaluation and reporting package
configs/             pilot.yaml = small data; base.yaml = full data
tests/               Automated data-integrity, model, loss, metric and resume checks
tools/               Progress inspection and notebook generation
notebooks/           Optional local walkthrough and Colab GPU workflow
Project_Documents/   The three current documents linked above
PTB-XL/              Raw data; kept local
data/                Prepared caches; kept local
experiments/         Checkpoints, logs, predictions and training status; kept local
outputs/             Figures/CSV tables; pilot CSV/PNG results are tracked
```

Future experiment configurations should use their own output directories.
The current small-data documents are in `Project_Documents/`; a future full-data
report goes to `Project_Documents/full_data/` so it cannot overwrite them.
There are no old incomplete reports mixed into the current reading material.

## Commands at a glance

Run PowerShell from the project root. Follow the guide for installation and data
preparation before training. A fresh clone does not include raw data, caches or
model weights.

```powershell
# Verify the code
.\.venv\Scripts\python -m pytest -q

# See available operations
.\.venv\Scripts\python -m tsa --help

# Inspect existing small-data progress; writes experiments/pilot/TRAINING_STATUS.md
.\.venv\Scripts\python tools/status.py --config configs/pilot.yaml

# Read existing comparisons, or regenerate the current report
.\.venv\Scripts\python -m tsa compare --config configs/pilot.yaml
.\.venv\Scripts\python -m tsa report --config configs/pilot.yaml
```

For fresh training, parameter changes, resume instructions and a larger cohort,
follow the [run guide](Project_Documents/CONFIGURATION_AND_RUN_GUIDE.md).
Completed runs are skipped automatically.

## Optional Colab GPU workflow

```powershell
.\.venv\Scripts\python -m tsa bundle --config configs/pilot.yaml
```

Put `outputs/pilot/colab_bundle.zip` in Google Drive at
`MyDrive/ptbxl_assignment/`, open `notebooks/colab_training.ipynb`, choose a GPU
runtime, and follow its cells. Results persist under
`MyDrive/ptbxl_assignment/pilot/`. Keep CPU/GPU runs in separate experiment paths.

The bundle includes code and the prepared small-data cache. It excludes raw data,
checkpoints, the attached PDF and the private manuscript comparison. The PDF remains
local and is excluded from Git. Building a bundle does not upload anything.

## Automated checks

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests tools
.\.venv\Scripts\python -m ruff format --check src tests tools
```

The 18 test cases use synthetic fixtures and cover patient isolation, chronological
history, missingness, training-only normalization, target isolation, model
shapes/gradients, contrastive masks, metrics, cache integrity and checkpoint resume.
GitHub Actions runs these checks when code is pushed; it does not train ECG models.
