# Patient-conditioned ECG forecasting

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
