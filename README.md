# Contrastive Patient-Stratified ECG Forecasts

**New to the project?** Start with the
[Configuration and Run Guide](Project_Documents/CONFIGURATION_AND_RUN_GUIDE.md).
Follow its local steps in order: install, configure the data location, test, prepare
data, train, and compare results. It explains why the small-data pilot exists and
shows how to run different models, change parameters, and move to the full dataset.

Forecast all 12 PTB-XL ECG leads at **500 Hz**, using **300 observed samples
(0.6 s)** to predict **150 future samples (0.3 s)**. Static conditioning uses
**age, sex, and strictly earlier diagnostic superclasses**: NORM, MI, STTC, CD, HYP.

The completed pilot compares LSTM and CNN-LSTM across four variants: ECG only,
demographics, patient history, and patient history with contrastive learning.
Persistence and repeated-window forecasts provide two simple baselines.

Start with the [pilot report](Project_Documents/pilot/ASSIGNMENT_REPORT.md) and
[results table](outputs/pilot/model_comparison.csv).

## Project layout

```text
src/tsa/             Data, models, training, evaluation, reporting, and CLI
configs/pilot.yaml   Default small-data experiment: up to 5 epochs
configs/base.yaml    Full-data experiment: up to 25 epochs
notebooks/           Local walkthrough and Colab GPU workflow
tests/               Data-integrity, model, loss, metric, and resume checks
tools/               Status inspection and notebook generation
PTB-XL/              Raw dataset (Git ignored)
data/                Full and pilot preprocessing caches (Git ignored)
experiments/         Configurations, checkpoints, predictions, metrics (Git ignored)
outputs/             Figures and comparisons; pilot CSV/PNG tracked, bundles ignored
Project_Documents/   Configuration/run guide, assignment reports, presentation notes
```

Its trained weights and results are not reused. Use `python -m tsa` for all workflows.

## Setup

Run PowerShell in this directory with Python 3.11+. Calling the virtual environment's
Python directly avoids PowerShell activation-policy issues.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q
```

Dependencies are declared once in `pyproject.toml`; `requirements.txt` installs the
package and development tools. `requirements-tested.txt` records tested local package
versions. For Linux/macOS use `.venv/bin/python`. Colab uses its preinstalled GPU
PyTorch; skip the CPU-wheel installation there.

## Run the pilot

The pilot and full preprocessing caches already exist locally. To reproduce the pilot:

```powershell
.\.venv\Scripts\python -m tsa preprocess --config configs/pilot.yaml
.\.venv\Scripts\python -u -m tsa run-experiments --config configs/pilot.yaml
.\.venv\Scripts\python -m tsa report --config configs/pilot.yaml
.\.venv\Scripts\python tools/status.py --config configs/pilot.yaml
```

Pilot preprocessing needs the full cache at `data/preprocessed/`. If starting from
raw files without that cache, first run the audit and preprocessing commands in the
full-data section below.

| Split | Patients | ECG recordings |
| --- | ---: | ---: |
| Training | 450 | 515 |
| Validation | 100 | 108 |
| Test | 100 | 116 |

Patients are sampled with seed 42 within their official folds; all records belonging
to each selected patient are kept. Age and waveform scalers are refitted on selected
training data only. Physical voltages are recovered from the full cache before
refitting, with possible float32 roundoff.

All eight pilot models and both baselines are complete. The lowest observed neural
MAE was **0.150113 mV** (LSTM with patient history); persistence achieved
**0.141546 mV**. These small-cohort, five-epoch results are preliminary and do not
establish a benefit from conditioning or contrastive learning.

Pilot artifacts are in `data/pilot/`, `experiments/pilot/`, `outputs/pilot/`, and
`Project_Documents/pilot/`. CLI commands default to `configs/pilot.yaml` when
`--config` is omitted.

## Train individual models and resume

```powershell
.\.venv\Scripts\python -u -m tsa train --model lstm --variant patient_history
.\.venv\Scripts\python -u -m tsa train --model cnn_lstm --variant patient_history_contrastive
.\.venv\Scripts\python -m tsa compare
```

| Variant | Static inputs | Dimensions |
| --- | --- | ---: |
| `ecg_only` | None | 0 |
| `demographics` | Age, sex, missingness flags | 4 |
| `patient_history` | Demographics, five prior diagnoses, history missing | 10 |
| `patient_history_contrastive` | Same inputs plus auxiliary contrastive training | 10 |

Interrupt with Ctrl+C, then rerun the same command. Training resumes from the last
completed epoch; completed runs are skipped. `best.pt` is chosen by validation
forecast MSE. `last.pt` also stores optimizer and RNG states. Only load trusted
checkpoints created by this project. A writer lock prevents conflicting preparation
or training jobs; keep the computer awake while training.

To inspect a selected checkpoint without training:

```powershell
.\.venv\Scripts\python -m tsa evaluate --model lstm --variant patient_history
```

## Increase the dataset later

Copy `configs/pilot.yaml`, increase `subset.patients`, and assign **new cache,
experiment, output, and document directories**. Then preprocess and train using
`--config` with the new file. With the same seed, increasing patient counts retains
the previously selected patients. Keep the test cohort fixed during development,
and select settings using validation results. Train larger-cohort comparisons from
fresh weights.

The full-data cache and stopped full-data checkpoints are preserved. To rebuild the
full cache or resume the original experiment explicitly:

```powershell
.\.venv\Scripts\python -u -m tsa audit --config configs/base.yaml
.\.venv\Scripts\python -u -m tsa preprocess --config configs/base.yaml
.\.venv\Scripts\python -u -m tsa run-experiments --config configs/base.yaml
```

Skip audit/preprocessing when reusing the existing valid cache. Official folds 1-8,
9, and 10 contain 17,418 / 2,183 / 2,198 recordings from disjoint patients. Use
`--root <path>` to override the raw-data directory or `--device cpu|cuda|auto` to
override device selection. Other settings live in YAML.

## Methods and results

LSTM uses two layers with hidden size 128. CNN-LSTM adds temporal convolutions with
kernels 7/5/3 and average pooling by two. Static features project to 32 dimensions
and fuse with the final dynamic state before a 256-unit forecasting head.

Contrastive variants augment observed inputs with jitter (standard deviation 0.01)
and global gains 0.95-1.05. NT-Xent uses temperature 0.1, excludes same-patient
negatives, and receives the dynamic representation before static fusion. The loss
is clean-input forecast MSE plus 0.1 times contrastive loss. Targets are unchanged.

AdamW uses learning rate 0.001, weight decay 0.0001, dropout 0.2, batch size 64,
and gradient clipping at 1.0. The pilot uses five epochs and patience three; full
training uses 25 epochs and patience five. These are initial settings, not tuned optima.

Preprocessing validates waveform dimensions, rate, units, lead order, and finite
values. Scalers fit training inputs only. Age sentinel 300 is missing. Diagnostic
inputs come from strictly earlier same-patient ECGs; current and equal-timestamp
labels are excluded. Missing history has its own flag. Current-record diagnoses
are used only for subgroup evaluation. Cache fingerprints prevent incompatible runs
from being compared.

Each experiment saves configuration/software versions, a feature manifest,
checkpoints, history, physical-mV predictions with IDs, and per-record/test metrics.
Figures include learning curves, lead/subgroup/horizon errors, and lowest-ID,
median-error, and worst-error twelve-lead overlays.

- MAE/RMSE use mV; MSE uses mV squared.
- R2 may be negative; correlation does not establish amplitude accuracy.
- Patient-macro MAE weights patients equally.
- Paired bootstrap intervals use 1,000 patient resamples for B-A; negative favors B.
- Diagnostic subgroups overlap; small groups and one training seed limit conclusions.

The report discusses the locally supplied paper and its protocol/metric inconsistencies.
The 0.6-second input can contain less than one beat. These forecasts do not establish
diagnostic accuracy, disease progression, or reduced clinical false alarms.

## Colab and notebooks

```powershell
.\.venv\Scripts\python -m tsa bundle --config configs/pilot.yaml
```

This creates `outputs/pilot/colab_bundle.zip` with code, settings, notebooks, tools,
and the pilot cache. Raw data, checkpoints, reports, the PDF, and extracted manuscript
metrics are excluded. No upload occurs automatically.

1. Put the ZIP in Google Drive at `MyDrive/ptbxl_assignment/`.
2. Open `notebooks/colab_training.ipynb` in Colab and select a T4 GPU runtime.
3. Run its cells to mount Drive, validate the cache, and train. Arrays stay on Colab
   local disk; results persist in `MyDrive/ptbxl_assignment/pilot/`.
4. Download that folder's `experiments`, `outputs`, and `documents` directories into
   local `experiments/pilot`, `outputs/pilot`, and `Project_Documents/pilot`.
   Keep CPU/GPU results separately, then regenerate the matching report.

`notebooks/assignment_walkthrough.ipynb` reads the pilot data, model shapes, results,
and figures. `tools/make_notebooks.py` rebuilds both notebooks. They default to the
pilot; full-data use requires changing their configuration/cache paths.

## Verification

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests tools
.\.venv\Scripts\python -m ruff format --check src tests tools
```

The 18 tests cover patient isolation, chronological diagnoses, train-only statistics,
target isolation, inverse normalization, model gradients, contrastive masks, metrics,
checkpoint resume, cache tampering, and repeatable/nested patient subsets.
