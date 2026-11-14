# Patient-conditioned ECG forecasting

Predict the next **3 seconds** of a twelve-lead ECG from **5 observed seconds**, using PTB-XL at **100 Hz**. Compare LSTM and CNN-LSTM, with optional age, sex, prior diagnoses and contrastive learning.

The main configuration is **4,500 training / 600 validation / 100 reserved test patients**. Use official folds 1–8 / 9 / 10. The test subset stays blocked during development.

The updated candidate uses hidden size 128, two LSTM layers, a 128-unit forecast head, learning rate 0.0005 and weight decay 0.0005. Training has a maximum of 60 epochs, validation-based learning-rate reduction, and early stopping after 10 non-improving epochs. The smaller output head reduces total parameters despite the larger recurrent encoder. These settings are candidates for validation, not measured performance improvements. The earlier run selected epochs 12/25 and produced weak heartbeat peaks; more epochs alone did not solve this.

## Run manually, one step at a time

The previous 1,500/300/100 configuration and measured results remain available through [configs/reference_1500.yaml](configs/reference_1500.yaml). Retrain for the expanded cohort before making accuracy claims.

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
  patients: {train: 4500, val: 600, test: 100}
```

Increase these counts, or change `mode` to `full` for all official patients. Cohort/window settings determine separate cache and result paths. Changing a filter/scaler gives a separate preprocessing cache. After changing the dataset, repeat preparation, audit, training, evaluation and report generation. Old scores do not become scores for the larger dataset.

Change `training.epochs`, `training.lr`, `training.batch_size` or the `model` settings to configure training. Use a new experiment name when changing model settings; existing runs are protected from accidental overwrites.

[configs/experiments.yaml](configs/experiments.yaml) contains optional named comparisons and parameter examples. It inherits the same dataset:

```powershell
python src/train.py --config configs/experiments.yaml --experiment lstm_early
python src/evaluate.py --config configs/experiments.yaml --experiment lstm_early --split val
```

Earlier presets and retired entry points still exist pending explicit cleanup approval; the active instructions use only the two configs above. See the cleanup section at the end of the run guide.

For controlled validation, `cnn_lstm_previous_settings` applies the previous model/training recipe to the expanded cohort. `cnn_lstm_1500_patients` applies the updated recipe to 1,500 training patients with the same 600 validation patients. `cnn_lstm_morphology` optionally adds derivative loss; it must be evaluated for both voltage error and peak preservation. These recipes are in `configs/experiments.yaml`. The previous 300-patient validation scores cannot be directly ranked against scores from 600 validation patients.

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

The earlier 500 Hz assignment and small 100 Hz study remain with their results in `outputs/pilot/` and `outputs/research_100hz_dev/`. The measured 1,500-patient report is preserved in `outputs/custom_100hz_5s_3s_ee477b14/ASSIGNMENT_REPORT.md`. Their scores are historical, not results for the expanded configuration. The earlier neural forecasts had poor R-peak preservation; larger training is an experiment, not a promised solution.

The private paper, raw data, caches, checkpoints and predictions remain excluded from repository publication. A reversible pre-cleanup source/document backup is local at `.cache/before_cleanup.zip`.
