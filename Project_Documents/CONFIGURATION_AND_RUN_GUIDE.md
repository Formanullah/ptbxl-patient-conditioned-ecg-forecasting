# How to configure and run the project locally

Follow Steps 1-8 in order. Section 9 shows how to train with different parameters.
Section 10 explains how to increase the dataset or run the full assignment.

All current reading material is together in this folder: this guide explains how
to run the project, [the assignment report](ASSIGNMENT_REPORT.md) explains the
measured results, and [the presentation notes](PRESENTATION_NOTES.md) explain the
architecture for class. Start from the repository README to choose a document.

## Why do we use “pilot”?

**Pilot means a small preliminary experiment.** It is the name of a configuration,
not a model or a required forecasting technique.

We created it because you asked to use less data when training took too long.
It lets you check the pipeline and try settings on your CPU before running a
larger experiment. The models, features and forecasting task are the same.

| | Small run: `configs/pilot.yaml` | Full run: `configs/base.yaml` |
| --- | --- | --- |
| Training | 450 patients / 515 ECGs | 17,418 ECGs |
| Validation | 100 patients / 108 ECGs | 2,183 ECGs |
| Testing | 100 patients / 116 ECGs | 2,198 ECGs |
| Maximum epochs | 5 | 25 |
| Early-stopping patience | 3 epochs | 5 epochs |
| Input → forecast | 300 → 150 samples at 500 Hz | Same |
| Architectures | LSTM and CNN-LSTM | Same |

Small-run findings are preliminary. You can use the full configuration directly
when you have enough time; small-data training is optional.

**Current implementation:** small-data preprocessing selects patients from a full
preprocessing cache. A fresh setup must therefore prepare that full cache once.
The small configuration reduces training time, but does not avoid initial full-data
preprocessing.

## Step 1. Open the project and install it

Run PowerShell in the project folder:

```powershell
cd D:\Study\HIS_Project\Project\ptbxl_assignment
```

If the working `.venv` already exists on this computer, skip installation.
For a fresh setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Confirm that installation worked:

```powershell
.\.venv\Scripts\python -m tsa --help
```

**Expected:** a list of commands, including `audit`, `preprocess`, `train` and
`compare`. Use this CLI for the pipeline. You do not need to run files inside
`src/tsa/` individually.

## Step 2. Configure the raw dataset location

The raw folder must contain:

```text
PTB-XL/
  ptbxl_database.csv
  scp_statements.csv
  records500/
    ... waveform .hea and .dat files ...
```

Check `data.root` in both `configs/base.yaml` and `configs/pilot.yaml`:

```yaml
data:
  root: PTB-XL
```

If the dataset is elsewhere, change only the root line, for example:

```yaml
  root: 'D:/Datasets/PTB-XL'
```

Keep the other fields under `data`. Use spaces for YAML indentation.
The root must point directly to the folder containing `ptbxl_database.csv`.

A GitHub clone includes code, documents and pilot CSV/PNG results. Raw ECG data,
preprocessing arrays, checkpoints and predictions must be available locally.

## Step 3. Test the code

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests tools
.\.venv\Scripts\python -m ruff format --check src tests tools
```

**Expected:** 18 passing tests, followed by successful lint/format checks. The tests
use synthetic ECG fixtures, so they do not need the full dataset.

If Windows denies access to pytest's temporary folder:

```powershell
.\.venv\Scripts\python -m pytest -q --basetemp .pytest-temp/run1 -p no:cacheprovider
```

This path is for disposable pytest files; pytest may clear its contents.

## Step 4. Prepare data

If valid `data/preprocessed/manifest.json` and `data/pilot/manifest.json` already
exist from our completed local work, skip to Step 5. Training verifies their hashes
and protocol before using them.

For a fresh setup, run:

```powershell
.\.venv\Scripts\python -u -m tsa audit --config configs/base.yaml
.\.venv\Scripts\python -u -m tsa preprocess --config configs/base.yaml
.\.venv\Scripts\python -m tsa preprocess --config configs/pilot.yaml
```

| Command | What it does | How to confirm |
| --- | --- | --- |
| `audit` | Checks every raw ECG's dimensions, 500 Hz rate, leads, mV units and finite values | `data/preprocessed/audit.json` has no failures |
| Full `preprocess` | Extracts ECG windows, builds static features and fits training-only normalization | `data/preprocessed/manifest.json` exists |
| Small `preprocess` | Selects whole patients and refits normalization on selected training records | `data/pilot/manifest.json` exists, with 515/108/116 records |

At 500 Hz, 300 observed samples are 0.6 seconds; the next 150 samples are the
0.3-second target. Every example contains all 12 leads.

Official folds 1-8 are training, 9 validation and 10 testing. Patients cannot occur
in multiple splits. Age uses training-median imputation with a missingness flag;
missing sex uses 0.5 plus a flag; unavailable prior diagnoses use zeros plus a
history-missing flag. Nonfinite ECG values stop preprocessing with a record list.

No model comparison is needed at this stage. First establish that the data is valid.

## Step 5. Create your own local training configuration

The original pilot models are already complete. Reusing their configuration prints
`SKIP completed`. To train fresh, copy the settings and choose new output folders:

```powershell
Copy-Item configs\pilot.yaml configs\local.yaml
```

Open `configs/local.yaml` and replace its `paths` section with:

```yaml
paths:
  experiments: experiments/local
  outputs: outputs/local
  documents: Project_Documents/local
```

Keep all other sections as copied, including `data.cache: data/pilot` and
`subset`. This reuses prepared data while saving your own models separately.

Use `--config configs/local.yaml` in the following commands. Omitting `--config`
would select the original `configs/pilot.yaml`.

## Step 6. Check training with a short real-data run

```powershell
.\.venv\Scripts\python -u -m tsa smoke --config configs/local.yaml --model lstm --variant patient_history
```

This runs at most two epochs with at most 128 records per split, checking loading,
training, checkpointing and evaluation.

**Expected:** epoch losses followed by `COMPLETE` and voltage metrics. Smoke results
are saved under `experiments/local/smoke/` and `outputs/local/smoke/`, separately
from normal comparisons. Use smoke to check execution, not to report model quality.

## Step 7. Train models

Choose A or B; you do not need both.

### A. Run all models automatically

```powershell
.\.venv\Scripts\python -u -m tsa run-experiments --config configs/local.yaml
```

This runs two baselines, trains all eight neural models, evaluates their selected
checkpoints and generates the report. Continue to Step 8.

### B. Run models one by one

First run the baselines:

```powershell
.\.venv\Scripts\python -m tsa baselines --config configs/local.yaml
```

Then train LSTM and CNN-LSTM with the same patient information:

```powershell
.\.venv\Scripts\python -u -m tsa train --config configs/local.yaml --model lstm --variant patient_history
.\.venv\Scripts\python -u -m tsa train --config configs/local.yaml --model cnn_lstm --variant patient_history
```

Each training command automatically selects the best validation checkpoint and
evaluates it on test data, saving metrics and plots.

For the complete eight-model assignment, also run:

```powershell
.\.venv\Scripts\python -u -m tsa train --config configs/local.yaml --model lstm --variant ecg_only
.\.venv\Scripts\python -u -m tsa train --config configs/local.yaml --model lstm --variant demographics
.\.venv\Scripts\python -u -m tsa train --config configs/local.yaml --model lstm --variant patient_history_contrastive
.\.venv\Scripts\python -u -m tsa train --config configs/local.yaml --model cnn_lstm --variant ecg_only
.\.venv\Scripts\python -u -m tsa train --config configs/local.yaml --model cnn_lstm --variant demographics
.\.venv\Scripts\python -u -m tsa train --config configs/local.yaml --model cnn_lstm --variant patient_history_contrastive
```

| Variant | Information used |
| --- | --- |
| `ecg_only` | ECG only |
| `demographics` | ECG + age/sex and missingness indicators: 4 static values |
| `patient_history` | Above + five strictly earlier diagnostic superclasses + history flag: 10 static values |
| `patient_history_contrastive` | Same inputs, plus the auxiliary contrastive training objective |

Run one training command at a time. To inspect progress from another window:

```powershell
.\.venv\Scripts\python tools/status.py --config configs/local.yaml
```

This prints progress and saves `experiments/local/TRAINING_STATUS.md` beside the
training logs. The main reading documents stay in `Project_Documents/`.

Press Ctrl+C to stop. Rerun the identical command to resume from the last completed
epoch. Finished runs are skipped.

## Step 8. Compare and read the results

After the chosen models finish:

```powershell
.\.venv\Scripts\python -m tsa compare --config configs/local.yaml
.\.venv\Scripts\python -m tsa report --config configs/local.yaml
```

The comparison command works with completed runs even before all eight finish.
The complete assignment needs eight neural runs and two baselines.

| File to open | What you learn |
| --- | --- |
| `outputs/local/model_comparison.csv` | MAE, RMSE and other metrics for completed runs |
| `outputs/local/model_comparison.png` | Model comparison chart |
| `outputs/local/paired_patient_bootstrap.csv` | Confidence intervals for pairs that have both runs |
| `Project_Documents/local/ASSIGNMENT_REPORT.md` | Methods, measured results and linked figures |
| `outputs/local/<model_variant>/learning_curves.png` | Training and validation behavior |
| `outputs/local/<model_variant>/forecast_lowest_id.png` | Predicted versus true ECG for a fixed example |

Compare the right pair for each question:

| Question | Compare |
| --- | --- |
| Does adding a CNN help? | LSTM versus CNN-LSTM with the same variant |
| Do age/sex help? | `ecg_only` versus `demographics`, same architecture |
| Does historical diagnosis help? | `demographics` versus `patient_history`, same architecture |
| Does contrastive learning help? | `patient_history` versus `patient_history_contrastive`, same architecture |
| Does learning beat simple forecasting? | Neural models versus persistence/repeated-window baselines |

Lower MAE/RMSE is better. MAE/RMSE are in mV; MSE is in mV squared.
Training/validation MSE uses normalized voltages and has a different scale from test
voltage MSE. Patient-macro MAE weights patients equally. In bootstrap comparisons,
B minus A below zero favors B; an interval crossing zero does not establish a clear
difference.

**Use validation MSE to choose parameters.** Report the held-out test results after
deciding the settings. Training automatically produces test metrics after each run,
so avoid choosing hyperparameters by repeatedly looking at those test results.

The existing small-data findings are in `outputs/pilot/`. Persistence has lower
MAE than every neural model there. These five-epoch findings are preliminary.

## 9. Run a model with different parameters

Example: compare an LSTM with hidden size 128 against hidden size 64, using the same
data and patient-history features.

The `local.yaml` configuration uses hidden size 128. Make another configuration:

```powershell
Copy-Item configs\local.yaml configs\local_h64.yaml
```

In `configs/local_h64.yaml`, replace the `model` section:

```yaml
model:
  hidden: 64
  depth: 2
  dropout: 0.2
  static_embedding: 32
```

Replace its output paths too:

```yaml
paths:
  experiments: experiments/local_h64
  outputs: outputs/local_h64
  documents: Project_Documents/local_h64
```

Keep `data.cache: data/pilot` and the complete `subset` section unchanged.
**Changing only model/training parameters does not require preprocessing again.**

Train:

```powershell
.\.venv\Scripts\python -u -m tsa train --config configs/local_h64.yaml --model lstm --variant patient_history
```

Inspect validation performance for both configurations:

```powershell
Get-Content experiments/local/lstm_patient_history/test_metrics.json | ConvertFrom-Json | Select-Object best_epoch,best_val_mse
Get-Content experiments/local_h64/lstm_patient_history/test_metrics.json | ConvertFrom-Json | Select-Object best_epoch,best_val_mse
```

Choose using validation MSE and learning curves. Normalized validation MSE is
comparable here because both runs use the same cache.

Generate each result table:

```powershell
.\.venv\Scripts\python -m tsa compare --config configs/local.yaml
.\.venv\Scripts\python -m tsa compare --config configs/local_h64.yaml
```

Open `outputs/local/model_comparison.csv` and
`outputs/local_h64/model_comparison.csv` side by side.
The CLI reads one experiment directory at a time. It does not automatically combine
different configuration directories or calculate confidence intervals between
arbitrary hyperparameter runs.

For another learning rate, dropout or batch size, follow the same pattern: copy a
config, change the setting, use new output folders, train the same model/variant,
and inspect validation performance. Change one setting at a time to understand its
effect.

### Available model and training parameters

These are YAML fields, not separate scripts.

| Field | Small-run default | Purpose |
| --- | --- | --- |
| `model.hidden` | 128 | LSTM hidden-state size |
| `model.depth` | 2 | Number of recurrent layers |
| `model.dropout` | 0.2 | Interlayer/head dropout; recurrent dropout requires depth > 1 |
| `model.static_embedding` | 32 | Patient-context representation size |
| `training.epochs` | 5 | Maximum training epochs |
| `training.patience` | 3 | Stop after this many unimproved validation epochs |
| `training.batch_size` | 64 | ECGs per batch |
| `training.lr` | 0.001 | AdamW learning rate |
| `training.weight_decay` | 0.0001 | Regularization |
| `training.gradient_clip` | 1.0 | Maximum gradient norm |
| `training.seed` | 42 | Initialization, training shuffle and augmentation seed |
| `training.device` | auto | CUDA when available, otherwise CPU |
| `training.threads` | 4 | CPU threads |
| `training.num_workers` | 0 | Loader processes; zero is reliable on Windows |
| `contrastive.weight` | 0.1 | Auxiliary loss weight |
| `contrastive.temperature` | 0.1 | NT-Xent similarity scaling |
| `contrastive.jitter_std` | 0.01 | Augmentation noise in normalized units |
| `contrastive.gain_min / gain_max` | 0.95 / 1.05 | Augmentation amplitude range |
| `evaluation.bootstrap_samples` | 1000 | Patient resamples for confidence intervals |

Contrastive fields affect `patient_history_contrastive` training only.
CNN kernels/channels and the 256-unit forecast head are fixed in
`src/tsa/models.py`; changing them requires code edits.

Existing checkpoints include a configuration signature. Changing epochs, learning
rate or other scientific settings in an already used experiment directory causes
an error. Keep original settings to resume; use new paths for changed settings and
fresh training.

CLI overrides are `--root <dataset-path>` and `--device cpu|cuda|auto`.
Change other settings in YAML; for example, `--lr` is not supported.

## 10. Increase data or use the full dataset

### Increase the small dataset

Copy `configs/local.yaml` to `configs/medium.yaml`, retaining all copied fields.
Edit these values:

| Field | New value |
| --- | --- |
| `data.cache` | `data/medium` |
| `subset.patients.train` | `1500` |
| `subset.patients.val` | `300` |
| `subset.patients.test` | Keep `100` |
| `training.epochs` | `15` |
| `training.patience` | `5` |
| `paths.experiments` | `experiments/medium` |
| `paths.outputs` | `outputs/medium` |
| `paths.documents` | `Project_Documents/medium` |

Keep `subset.source_cache: data/preprocessed` and `subset.seed: 42`.
The counts select patients, not recordings. The same subset seed gives nested
patient selections when counts increase. Keep the test cohort fixed while developing.

```powershell
.\.venv\Scripts\python -m tsa preprocess --config configs/medium.yaml
.\.venv\Scripts\python -u -m tsa run-experiments --config configs/medium.yaml
```

Changing the cohort requires preprocessing: selected records and training
normalization change. Train new models from fresh weights.

### Run the full dataset

`configs/base.yaml` has no `subset` section. It uses the full cache from Step 4.
For a fresh full-data experiment:

```powershell
Copy-Item configs\base.yaml configs\full_local.yaml
```

In the copied config, replace `paths` with:

```yaml
paths:
  experiments: experiments/full_local
  outputs: outputs/full_local
  documents: Project_Documents/full_local
```

Retain the other copied sections, correct `data.root` if necessary, then run:

```powershell
.\.venv\Scripts\python -u -m tsa run-experiments --config configs/full_local.yaml
```

To deliberately resume the earlier stopped full-data work instead, use
`--config configs/base.yaml` with its original settings and paths.

### Change the ECG window

| Field | Default | Meaning |
| --- | --- | --- |
| `data.sampling_rate` | 500 | Samples per second; supports 100 or 500 |
| `data.history` | 300 | Observed samples |
| `data.horizon` | 150 | Future samples |
| `data.start` | 0 | Starting sample within the ten-second record |

Window/sampling changes require a new matching full cache, a new subset cache if
used, and new experiment paths. The windows must fit:
`start + history + horizon <= 10 * sampling_rate`.
Keep the agreed 500 Hz / 300-to-150 protocol for the current assignment comparison.

## 11. Find the files for a model

For `lstm` + `patient_history` under `configs/local.yaml`:

```text
experiments/local/lstm_patient_history/
  config.json              Settings, software versions, device, parameter counts
  feature_manifest.json    Feature order and preprocessing information
  history.csv              Train/validation losses for each epoch
  best.pt                  Validation-selected weights
  last.pt                  Resume state: weights, optimizer and random states
  test_metrics.json        Test metrics, best epoch and validation MSE
  predictions.npz          Forecasts, targets and ECG/patient IDs
  per_record_metrics.csv   Errors per test ECG

outputs/local/lstm_patient_history/
  learning_curves.png
  per_lead_metrics.csv, per_lead_errors.png
  subgroup_metrics.csv, subgroup_errors.png
  horizon_metrics.csv, horizon_error.png
  forecast_lowest_id.png
  forecast_median_error.png
  forecast_worst_error.png
```

Inspect an existing selected model without training:

```powershell
.\.venv\Scripts\python -m tsa evaluate --config configs/local.yaml --model lstm --variant patient_history
```

This returns existing test metrics if present; otherwise, it evaluates the saved
best checkpoint. It requires locally available weights.

## 12. Which script does what?

| File | Responsibility | Command |
| --- | --- | --- |
| `src/tsa/__main__.py` | CLI routing and writer lock | `python -m tsa ...` |
| `src/tsa/config.py` | YAML, feature definitions, protocol signatures | Used automatically |
| `src/tsa/data.py` | Audit, static features, normalization, caches and DataLoaders | `audit`, `preprocess` |
| `src/tsa/models.py` | LSTM, CNN-LSTM, static fusion, augmentations and contrastive loss | Used by training |
| `src/tsa/training.py` | Optimization, checkpoints, resume, inference and baselines | `smoke`, `train`, `evaluate`, `baselines`, `benchmark`, `run-experiments` |
| `src/tsa/evaluation.py` | Metrics, plots, comparisons and confidence intervals | Used after training; `compare` |
| `src/tsa/reporting.py` | Report, presentation notes and Colab ZIP | `report`, `bundle` |
| `tests/test_pipeline.py` | Automated integrity/model/metric/resume tests | `python -m pytest -q` |
| `tools/status.py` | Prints progress and writes its Markdown document | `python tools/status.py --config ...` |
| `tools/make_notebooks.py` | Recreates both notebooks; overwrites their contents | Optional developer tool |
| `notebooks/assignment_walkthrough.ipynb` | Interactive data/model/results explanation | Open in VS Code/Jupyter |
| `notebooks/colab_training.ipynb` | GPU training with the same package | Optional; see README's Colab instructions |
| `.github/workflows/ci.yml` | Tests and lint on pushes/PRs | Automatic on GitHub; does not train models |

The optional timing command is
`python -m tsa benchmark --config configs/local.yaml`.
It estimates training time from short passes, excluding validation/plotting
overhead. It does not save model weights or evaluate test performance.

The `report` command reads existing results. It does not train missing models or
recreate all per-model plots. It writes the report and presentation notes only to
the configured documents directory. Training progress is stored beside experiment
logs, not among the reading documents.

Read the [completed small-data report](ASSIGNMENT_REPORT.md) and
[presentation notes](PRESENTATION_NOTES.md) for the existing measured results.

## 13. Common problems

| Problem | Action |
| --- | --- |
| `No module named tsa` | Install `requirements.txt` with the same `.venv` Python |
| Missing CSV/waveform | Check `data.root` and the Step 2 folder layout |
| Missing manifest | Prepare full cache, then small cache as in Step 4 |
| Cache mismatch/modified cache | Check settings; rebuild the matching cache and use fresh experiment paths |
| `SKIP completed` | Use Step 5 to create a fresh configuration with new output paths |
| Configuration changed | Restore original settings to resume, or train fresh under new paths |
| Another job active | Wait for the writer; inspect `tools/status.py` |
| CUDA unavailable | Use `--device cpu` |
| CUDA out of memory | Reduce batch size in a fresh experiment configuration |
| No selected checkpoint | Train that model/variant first; GitHub does not contain weights |

The physical `.pipeline.lock` file can remain while idle. Its presence alone does
not mean training is running. Keep the computer awake during long local training.
