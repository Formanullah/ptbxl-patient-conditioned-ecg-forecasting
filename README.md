# Patient-conditioned ECG forecasting

Compare LSTM and CNN-LSTM forecasts of 12-lead PTB-XL ECGs using age, sex and earlier
diagnostic history. Input: 300 samples (0.6 seconds). Forecast: 150 samples
(0.3 seconds), at 500 Hz.

Run each step yourself. Settings live in YAML files; each script performs one
chosen operation.
All project scripts and model code are directly in `src/`. Each stage script
contains its implementation; there are no duplicate root launchers or nested package.

## Project structure

```text
ptbxl_assignment/
  src/
    preprocessing.py    Data preparation and DataLoaders
    train.py            Training implementation
    evaluate.py         Evaluation, metrics and plots
    compare.py          Model comparisons
    report.py           Report generation
    models.py           LSTM and CNN-LSTM definitions
    config.py           YAML settings and common constants
    run_tests.py        Software test runner
    status.py           Inspect saved progress
  configs/              Edit local.yaml
  tests/                Test cases
  notebooks/            Local walkthrough and Colab notebook
  Project_Documents/    Run guide, report and presentation notes
  PTB-XL/               Raw dataset
  data/                 Prepared arrays
  experiments/          Model checkpoints, logs and predictions
  outputs/              Figures and comparison tables
  .cache/               Disposable test/tool files; ignored by Git
```

Run commands from the project root using `python src/<script>.py`. A model's files
go inside the configured experiment directory. Temporary tool files stay in
`.cache/`. `.venv/` holds your Python environment and `.github/` holds the manual
GitHub check definition.

## 1. Activate your environment

```powershell
cd D:\Study\HIS_Project\Project\ptbxl_assignment
.\.venv\Scripts\Activate.ps1
```

For a fresh setup, create the environment first, then activate and install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If PowerShell blocks activation, use
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and retry activation.
After activation, use `python` normally in that terminal.

## 2. Choose and edit a configuration

`configs/pilot.yaml` uses a small patient subset and five epochs.
`configs/base.yaml` uses all data and up to 25 epochs.
“Pilot” simply means the small preliminary experiment requested to reduce CPU time.

**Edit `configs/local.yaml` for your work.** It is the default for the scripts and
already uses the small-data settings with separate output paths. Set your raw
`data.root` and review:

```yaml
paths:
  experiments: experiments/local
  outputs: outputs/local
  documents: Project_Documents/local
```

Change `model.hidden`, `training.lr`, `training.batch_size` and
`training.epochs` in the same file as needed. Use new output paths for each changed
experiment. Keep `data.cache: data/pilot` when reusing the prepared small dataset.

## 3. Prepare data

Skip this step when the matching caches already exist. For a fresh setup, check
`data.root` in both original configurations, then run:

```powershell
python src/preprocessing.py --audit --config configs/base.yaml
python src/preprocessing.py --config configs/base.yaml
python src/preprocessing.py
```

The small-data preparation currently selects patients from the full cache, which
must be prepared once. Preprocessing is shared by both models.

## 4. Train and evaluate one model at a time

```powershell
python src/train.py --model lstm --variant patient_history
python src/evaluate.py --model lstm --variant patient_history

python src/train.py --model cnn_lstm --variant patient_history
python src/evaluate.py --model cnn_lstm --variant patient_history
```

Training saves checkpoints and validation metrics. Evaluation saves test metrics,
predictions and figures. Run evaluation after choosing settings using validation.

To continue an interrupted run, explicitly add `--resume` to its training command.
Starting the same saved run without `--resume` produces a message asking you to
resume or choose fresh output paths.

## 5. Compare and write the report

```powershell
python src/evaluate.py --baselines
python src/compare.py
python src/report.py
```

Open `outputs/local/model_comparison.csv` and
`Project_Documents/local/ASSIGNMENT_REPORT.md`.
Compare LSTM and CNN-LSTM with the same variant and data.

## Scripts

| Script | What you choose to run |
| --- | --- |
| `src/preprocessing.py` | Prepare arrays; `--audit` checks raw recordings |
| `src/train.py` | Train one `--model` and `--variant`; `--resume` continues its saved state |
| `src/evaluate.py` | Evaluate one saved model, or `--baselines` for simple forecasts |
| `src/compare.py` | Compare already evaluated runs |
| `src/report.py` | Generate documents from available results |
| `src/run_tests.py` | Run automated software tests when you choose |
| `src/status.py` | Inspect progress without training anything |

Use `python src/train.py --help` for options. To use another YAML file, add
`--config configs/your_experiment.yaml` to a command.
Variants: `ecg_only`, `demographics`, `patient_history`,
`patient_history_contrastive`. Repeat training and evaluation manually for the
variants you want. There is no “run everything” command.

## Tests and documentation

```powershell
python src/run_tests.py -q
```

Use `src/run_tests.py` rather than `pytest.py`, which would conflict with pytest imports.
GitHub checks can be started manually from the repository's Actions tab.

| Document | Purpose |
| --- | --- |
| [Run Guide](Project_Documents/CONFIGURATION_AND_RUN_GUIDE.md) | Detailed configuration, manual steps, parameter comparisons, resume and troubleshooting |
| [Assignment Report](Project_Documents/ASSIGNMENT_REPORT.md) | Existing small-data methods, metrics, figures and literature comparison |
| [Presentation Notes](Project_Documents/PRESENTATION_NOTES.md) | CNN/LSTM architecture and training explanation for class |

The existing small-data results include eight neural runs and two baselines.
The best neural MAE is 0.150113 mV; persistence achieves 0.141546 mV.
These preliminary results do not establish a benefit from conditioning or
contrastive learning. See the [results table](outputs/pilot/model_comparison.csv).

Raw data, caches, checkpoints and predictions stay local. The attached paper PDF
is excluded from Git and Colab transfers.
