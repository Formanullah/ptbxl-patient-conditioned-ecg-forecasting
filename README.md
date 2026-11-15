<<<<<<< HEAD
﻿# Patient-conditioned ECG forecasting

Predict the next **3 seconds** of a twelve-lead ECG from **5 observed seconds**, using PTB-XL at **100 Hz**. Compare LSTM and CNN-LSTM, with optional age, sex, prior diagnoses and contrastive learning.

The main configuration is **1,500 training / 300 validation / 100 reserved test patients**. Use official folds 1–8 / 9 / 10. The test subset stays blocked during development.

## Run manually, one step at a time

These are instructions for you to run. The cleanup changes have **not been executed or tested**, as requested.

From the project folder, create an environment if needed, then activate it once:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `.venv` already exists, start with activation. No package installation or `python -m tsa` command is needed.

```powershell
# 1. Check software, then prepare and audit the selected patients.
python src/run_tests.py -q
python src/preprocessing.py
python src/leakage_audit.py

# 2. Train one model, then explicitly evaluate it on validation patients.
python src/train.py --model lstm --variant patient_history
python src/evaluate.py --model lstm --variant patient_history --split val

# 3. Train and evaluate the architecture comparison.
python src/train.py --model cnn_lstm --variant patient_history
python src/evaluate.py --model cnn_lstm --variant patient_history --split val

# 4. Evaluate baselines and refresh the report.
python src/evaluate.py --baselines --split val
python src/report.py
```

All commands default to `configs/training.yaml`. Nothing runs every model automatically. Training saves checkpoints, epoch metrics and learning curves; evaluation saves waveform/peak metrics and figures; reporting reads saved results.

## Change the dataset or model

Edit [configs/training.yaml](configs/training.yaml). This is the one main configuration.

```yaml
subset:
  mode: custom
  seed: 42
  patients: {train: 1500, val: 300, test: 100}
```

Increase these counts, or change `mode` to `full` for all official patients. Cohort/window settings determine separate cache and result paths. Changing a filter/scaler gives a separate preprocessing cache. After changing the dataset, repeat preparation, audit, training, evaluation and report generation. Old scores do not become scores for the larger dataset.

Change `training.epochs`, `training.lr`, `training.batch_size` or the `model` settings to configure training. Use a new experiment name when changing model settings; existing runs are protected from accidental overwrites.

[configs/experiments.yaml](configs/experiments.yaml) contains optional named comparisons and parameter examples. It inherits the same dataset:

```powershell
python src/train.py --config configs/experiments.yaml --experiment lstm_early
python src/evaluate.py --config configs/experiments.yaml --experiment lstm_early --split val
```

Earlier presets and retired entry points still exist pending explicit cleanup approval; the active instructions use only the two configs above. See the cleanup section at the end of the run guide.

## Project layout

```text
src/                 All Python scripts and reusable functions
configs/             training.yaml and optional experiments.yaml
tests/              Software and leakage checks
Project_Documents/   Run guide, report, feature dictionary, audit and presentation notes
PTB-XL/              Raw dataset, retained locally
data/               Generated caches and manifests, ignored by Git
experiments/         Checkpoints, configurations, histories and predictions, ignored
outputs/             Tables and figures, separated by cohort
```

## Which file should I read?

- [Run guide](Project_Documents/CONFIGURATION_AND_RUN_GUIDE.md): script responsibilities, commands, parameter changes, comparisons, resume and final testing.
- [Assignment report](Project_Documents/ASSIGNMENT_REPORT.md): the current cohort's status and measured results; refresh with `python src/report.py`.
- [Data dictionary](Project_Documents/DATA_DICTIONARY.md): features, missing values, shapes and units.
- [Leakage audit](Project_Documents/DATA_LEAKAGE_AUDIT.md): controls, historical evidence and checks still to run after cleanup.
- [Presentation notes](Project_Documents/PRESENTATION_NOTES.md): explain the current methods and report results.

The earlier 500 Hz assignment and small 100 Hz study remain with their results in `outputs/pilot/` and `outputs/research_100hz_dev/`. Their scores are historical, not results for the 1,500-patient training configuration. The earlier neural forecasts had poor R-peak preservation; larger training is an experiment, not a promised solution.

The private paper, raw data, caches, checkpoints and predictions remain excluded from repository publication. A reversible pre-cleanup source/document backup is local at `.cache/before_cleanup.zip`.
=======
﻿# PTB-XL waveform forecasting

Observe 7 seconds and predict the remaining 3 seconds of a native 100-Hz,
twelve-lead ECG. This is within-record waveform forecasting, not clinical
longitudinal prediction.

## Setup and quick check

Use Python 3.11+ and install `requirements.txt` in a virtual environment. The
existing local `.venv` can be used. The dataset root must contain
`ptbxl_database.csv`, `scp_statements.csv`, `SHA256SUMS.txt`, and `records100/`.

```sh
python -m pip install -r requirements.txt
python src/run_tests.py -q
python src/pipeline.py prepare
python src/pipeline.py audit
python src/pipeline.py smoke
python src/pipeline.py report
```

`configs/config.yaml` is the active configuration. Pilot mode defaults to 300
training and 75 validation patients, retaining all their ECGs, for two epochs.
Pilot/synthetic scores are software diagnostics, not research evidence. Test
waveforms are not prepared or loaded in this workflow.

For the separate 5s→5s alternative, copy the configuration, set `data.history`
and `data.horizon` to 500, and choose a new `study_id`. Use `--config` to run
that file. Its caches, checkpoints, and results cannot be pooled with 7s→3s.

For a tiny dataset-free software check:

```sh
python src/pipeline.py smoke --fixture-dir .cache/synthetic-check
```

## Research stages

```sh
python src/pipeline.py screen --mode medium
python src/pipeline.py tune --mode medium
python src/pipeline.py features --mode medium
python src/pipeline.py full --mode full
python src/pipeline.py final-test --mode full --enable-final-test
python src/pipeline.py report --mode full
python src/pipeline.py export --mode full
```

Screening uses 1,500 training patients and all official validation patients, with
60-epoch controlled model/filter and F0/F1 comparisons. Tuning uses the predefined
learning rates. Feature selection is a separate expanded-metadata extension using
training-only mRMR and grouped validation permutation importance; all 12 leads stay.

Full training uses all official train/validation patients and four matched
120-epoch conditions: LSTM/CNN-LSTM × raw/causally filtered history. Each filter
pair shares features, hyperparameters, raw-input scaler and raw-mV target.
Full training requires CUDA unless `allow_full_cpu` is explicitly enabled.
The final-test command requires the frozen four-condition selection; its ledger
reuses completed results and resumes interrupted conditions without retuning.
ARIMA and contrastive training are not part of this primary workflow.

Use `--resume` for interrupted runs. A completed run can be reused only after its
scientific signature matches. Changed cohort/model/source/settings require a new
study or experiment identity. Raw-source caches live under `data/study_v3`; new
results, checkpoints, figures and manifests live under `results/<study_id>/`.

## Notebook: local or Colab

Open `notebooks/PTBXL_Full_Colab_GPU.ipynb` in Jupyter or VS Code and select
the Python kernel with this project's dependencies. On Windows, use
`.venv\Scripts\python.exe` if you already have the project environment. The
notebook finds the project root automatically when you start it there or under
`notebooks/`. Otherwise set `LOCAL_PROJECT_DIR` in the first cell. Set
`LOCAL_DATA_ROOT` if the dataset is outside the project's `PTB-XL/` folder.
Local runs do not mount Google Drive. By default they save caches in
`data/study_v3/` and outputs in `results/<study_id>/`.

In Colab, set `DRIVE_PROJECT` or upload the source ZIP and choose Drive data
or the official PhysioNet download of `records100` only. Both environments
start in pilot mode. The first cell controls patient counts for pilot and
medium screening; use a new `STUDY_ID` when changing the cohort. Section 4
shows the resulting ECG and patient counts, including reserved test metadata.
Full mode uses the official full cohorts and requires CUDA unless explicitly
enabled on CPU. Results are exported as `PTBXL_Research_Results.zip`.

## Results and verification

The [CPU pilot report](docs/RESULTS.md) covers six two-epoch runs on 339 training
and 81 validation ECGs. Fold 10 remained unopened. All 26 acceptance tests
passed, and an isolated kernel executed all 11 notebook code cells on synthetic
data. Synthetic checks also exercised screening, tuning, both feature-selection
methods, and final-test locking. A checksum-verified official download and WFDB
decode were tested for one 100-Hz record.

Real-data screening, tuning, feature selection, the four 120-epoch GPU runs, and
locked final scoring are pending. Pilot and synthetic scores do not rank the
research conditions.

The shared CLI is `src/pipeline.py`; the notebook calls the same experiment code.
See [methodology](docs/METHODOLOGY.md), [equations](docs/EQUATIONS.md), and the
[data dictionary](docs/DATA_DICTIONARY.md). The [historical archive](archive/legacy_2ad3e68/README_ARCHIVE.md)
explains earlier protocols and which old reports lack supporting artifacts.
>>>>>>> origin/work/manual-src-workflow
