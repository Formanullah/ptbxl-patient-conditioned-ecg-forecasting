# Manual configuration and run guide

All project code is directly in `src/`. Run commands from the project root using
`python src/<script>.py`. Activate the environment once per terminal, edit one YAML
configuration, then choose which model and operation to run.

## 1. What “pilot” means

`configs/pilot.yaml` selects a small patient subset to reduce CPU training time.
It is the same forecasting task and the same models as the full-data experiment.

| Configuration | Training / validation / test ECGs | Maximum epochs |
| --- | --- | --- |
| `pilot.yaml` | 515 / 108 / 116 | 5 |
| `base.yaml` | 17,418 / 2,183 / 2,198 | 25 |

The small split contains 450/100/100 patients. All records of a selected patient are
kept, with patients separated between splits. Small-data training is optional;
you can use the full dataset when you have enough time.

Current implementation: preparing the small cache requires a matching full cache
once. Smaller training does not avoid initial full-data preprocessing.

## 2. Install and activate

Open PowerShell at the project root:

```powershell
cd D:\Study\HIS_Project\Project\ptbxl_assignment
```

For a fresh setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For an existing setup, only activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and
retry. The prompt should show `(.venv)`. Use `python` normally afterward.

## 3. Configure one experiment

Open the ready-made `configs/local.yaml`. All scripts use it by default.
It starts with small-data settings and separate local output paths. Keep its
sections and edit the values you need. An explicit `--config` selects another YAML.

| Field | Purpose | Small-run value |
| --- | --- | --- |
| `data.root` | Folder containing raw metadata and waveform directories | `PTB-XL` |
| `data.cache` | Prepared arrays to reuse | `data/pilot` |
| `data.sampling_rate` | ECG samples per second | `500` |
| `data.history` | Observed samples | `300` |
| `data.horizon` | Future samples | `150` |
| `data.start` | Starting sample | `0` |
| `subset.source_cache` | Full cache used to select the small cohort | `data/preprocessed` |
| `subset.seed` | Patient-selection seed | `42` |
| `subset.patients.train/val/test` | Patient counts | `450/100/100` |
| `model.hidden` | LSTM hidden size | `128` |
| `model.depth` | Recurrent layers | `2` |
| `model.dropout` | Recurrent interlayer/head dropout | `0.2` |
| `model.static_embedding` | Static feature embedding width | `32` |
| `training.seed` | Initialization, shuffle and augmentation seed | `42` |
| `training.epochs` | Maximum epochs | `5` |
| `training.patience` | Stop after this many unimproved validation epochs | `3` |
| `training.batch_size` | Recordings per batch | `64` |
| `training.lr` | AdamW learning rate | `0.001` |
| `training.weight_decay` | Regularization | `0.0001` |
| `training.gradient_clip` | Maximum gradient norm | `1.0` |
| `training.device` | `auto`, `cpu` or `cuda` | `auto` |
| `training.threads` | CPU threads | `4` |
| `training.num_workers` | DataLoader worker processes | `0` |
| `contrastive.weight` | Additional loss weight | `0.1` |
| `contrastive.temperature` | NT-Xent similarity scaling | `0.1` |
| `contrastive.jitter_std` | Noise in normalized units | `0.01` |
| `contrastive.gain_min/gain_max` | ECG augmentation gain range | `0.95/1.05` |
| `evaluation.bootstrap_samples` | Patient resamples for comparison intervals | `1000` |

The default local output paths are:

```yaml
paths:
  experiments: experiments/local
  outputs: outputs/local
  documents: Project_Documents/local
```

Use new paths for each changed experiment. Existing checkpoints require the same
scientific settings, including epochs. Paths, CPU threads, worker count and device
can change for portability; results can still vary across hardware.

## 4. Prepare data

Raw data must contain `ptbxl_database.csv`, `scp_statements.csv` and `records500/`
under `data.root`. Correct the root in the full and local configurations.

For fresh data:

```powershell
python src/preprocessing.py --audit --config configs/base.yaml
python src/preprocessing.py --config configs/base.yaml
python src/preprocessing.py --config configs/local.yaml
```

The first checks raw data, the second builds the full cache, and the third prepares
your small subset. If the matching caches already exist, skip preparation.

Checks cover shape, sampling rate, units, leads, finite ECG values and patient
isolation. Invalid records are listed explicitly. Normalization uses training input
windows only. Age/sex and strictly earlier diagnoses become static inputs with
missingness flags. The manifest stores the protocol, identifiers, statistics and
file hashes.

There is no model choice for preprocessing: both architectures use the same arrays.
At 500 Hz, the input covers 0.6 seconds and the target the next 0.3 seconds.

## 5. Test the software when needed

```powershell
python src/run_tests.py -q
```

Tests use synthetic ECG fixtures. They check patient isolation, chronology,
normalization, model gradients, contrastive masks, metrics, explicit resume and
the separation between training and evaluation.

Use `src/run_tests.py`, rather than `pytest.py`, to avoid shadowing pytest.
You can also run `pytest -q` directly after activating the environment.

Disposable test files are stored in `.cache/test-temp`, pytest's cache in
`.cache/pytest`, and Ruff's cache in `.cache/ruff`. These paths are configured in
`pyproject.toml`, keeping temporary tool folders together. They are ignored by Git.
The test-temp directory is reserved for pytest, which clears it between runs.

## 6. Train one model

```powershell
python src/train.py --config configs/local.yaml --model lstm --variant patient_history
```

This uses training and validation loaders. It saves the selected weights,
resumable state, validation summary and learning curve. It does not run test
inference, another model, comparison or reporting.

Inspect:

```text
experiments/local/lstm_patient_history/
  config.json
  feature_manifest.json
  history.csv
  best.pt
  last.pt
  training_metrics.json
```

`best.pt` is chosen by validation forecast MSE. `training_metrics.json` includes
`best_val_mse`, `best_epoch` and runtime. Training stops at the configured epoch
limit or early-stopping patience.

For a quick check, copy the configuration, set `training.epochs: 2` and
`training.patience: 2`, and use separate output paths. There is no hidden smoke
mode or automatic experiment matrix.

## 7. Evaluate that model separately

After choosing settings using validation performance:

```powershell
python src/evaluate.py --config configs/local.yaml --model lstm --variant patient_history
```

This runs test inference with the selected checkpoint and writes:

```text
experiments/local/lstm_patient_history/
  test_metrics.json
  predictions.npz
  per_record_metrics.csv

outputs/local/lstm_patient_history/
  per_lead_metrics.csv, per_lead_errors.png
  subgroup_metrics.csv, subgroup_errors.png
  horizon_metrics.csv, horizon_error.png
  forecast_lowest_id.png
  forecast_median_error.png
  forecast_worst_error.png
```

Calling evaluation again recomputes the selected model's test metrics and figures.
No model is trained by this command.

Now choose CNN-LSTM explicitly:

```powershell
python src/train.py --config configs/local.yaml --model cnn_lstm --variant patient_history
python src/evaluate.py --config configs/local.yaml --model cnn_lstm --variant patient_history
```

## 8. Compare evaluated models

First evaluate simple baselines, then request the comparison:

```powershell
python src/evaluate.py --baselines --config configs/local.yaml
python src/compare.py --config configs/local.yaml
```

Open `outputs/local/model_comparison.csv` or `model_comparison.png`.
Only evaluated runs appear. Training a model alone does not add test results.

| Question | Pair to compare |
| --- | --- |
| Does a CNN help? | LSTM versus CNN-LSTM, same variant |
| Do age and sex help? | `ecg_only` versus `demographics`, same architecture |
| Does diagnostic history help? | `demographics` versus `patient_history`, same architecture |
| Does contrastive learning help? | `patient_history` versus `patient_history_contrastive`, same architecture |
| Does learning beat simple forecasting? | Neural forecasts versus persistence/repeated-window baselines |

Repeat Steps 6-7 yourself for each architecture/variant you choose.
Both architectures accept:

```text
ecg_only
demographics
patient_history
patient_history_contrastive
```

Eight neural comparisons means two architectures times four variants.
The confidence-interval table contains only pairs with both evaluated runs.

Lower MAE/RMSE is better. MAE/RMSE are in mV and MSE in mV squared. Train/validation
MSE uses normalized units. Bootstrap differences are B minus A; negative favors B.
An interval crossing zero does not establish a clear difference.

## 9. Generate documents when you choose

```powershell
python src/report.py --config configs/local.yaml
```

This writes the report and presentation notes to `Project_Documents/local/`, using
available evaluation artifacts. It does not train or evaluate missing models.
Missing comparisons are marked incomplete.

The existing report and notes in the root of `Project_Documents/` describe the
completed original small-data runs; your new local report stays separate.

## 10. Try different parameters

Example: change hidden size from 128 to 64 without changing the data.

```powershell
Copy-Item configs/local.yaml configs/local_h64.yaml
```

In the copied file, change `model.hidden: 64` and replace the three output paths
with `experiments/local_h64`, `outputs/local_h64` and
`Project_Documents/local_h64`. Keep all other fields and the same cache.

```powershell
python src/train.py --config configs/local_h64.yaml --model lstm --variant patient_history
```

Compare validation performance before choosing settings:

```powershell
Get-Content experiments/local/lstm_patient_history/training_metrics.json | ConvertFrom-Json | Select-Object best_epoch,best_val_mse
Get-Content experiments/local_h64/lstm_patient_history/training_metrics.json | ConvertFrom-Json | Select-Object best_epoch,best_val_mse
```

These validation values are comparable because the runs share the same cache.
Choose settings using validation, then explicitly evaluate the chosen run.
For older existing runs without `training_metrics.json`, inspect `history.csv`.

Follow the same copy/change/train pattern for learning rate, dropout or batch size.
Change one setting at a time to understand its effect. Model/training changes reuse
the cache; changing the cohort or ECG window requires new preprocessing.

The comparison script reads one configured experiment directory. It does not
combine arbitrary hyperparameter directories. You can open their result tables
side by side after evaluation.

## 11. Resume deliberately

Press Ctrl+C to stop. To continue the exact same model/configuration:

```powershell
python src/train.py --resume --config configs/local.yaml --model lstm --variant patient_history
```

Resume restores weights, optimizer state, shuffle/augmentation randomness and
history from the last completed epoch. Work in an interrupted epoch is repeated.

Without `--resume`, an existing saved run produces an explicit error. To start
fresh, choose new output paths. Changing the epoch limit or other scientific
settings in the original run is rejected; copy the config for a fresh experiment.

If the original run has already reached its epoch/early-stopping limit, explicit
resume performs no additional epochs. You can evaluate its selected checkpoint.

Inspect progress:

```powershell
python src/status.py --config configs/local.yaml
```

Progress is saved beside experiment logs as `TRAINING_STATUS.md`.
A writer lock prevents concurrent preparation/training/evaluation from colliding.

## 12. Increase the data

For a medium-sized cohort, copy the local config and change:

| Field | Example |
| --- | --- |
| `data.cache` | `data/medium` |
| `subset.patients.train/val/test` | `1500/300/100` |
| `training.epochs / patience` | `15 / 5` |
| `paths.experiments` | `experiments/medium` |
| `paths.outputs` | `outputs/medium` |
| `paths.documents` | `Project_Documents/medium` |

Keep the same subset seed and test cohort during development.
Run preprocessing for the new cohort, then train/evaluate each chosen model.

For all data, copy `configs/base.yaml` to `configs/full_local.yaml` and set new
output paths. It has no `subset` section and reuses the full prepared cache:

```powershell
python src/train.py --config configs/full_local.yaml --model lstm --variant patient_history
python src/evaluate.py --config configs/full_local.yaml --model lstm --variant patient_history
```

Changing the ECG window/rate requires a new matching full cache and, if used, a
new subset cache. Windows must fit within ten seconds:
`start + history + horizon <= 10 * sampling_rate`.
Keep 500 Hz and 300-to-150 windows for the agreed assignment comparison.

## 13. Script and file reference

| File | Responsibility |
| --- | --- |
| `src/preprocessing.py` | Prepare data; `--audit` checks raw recordings |
| `src/train.py` | Train one selected model; explicit `--resume` |
| `src/evaluate.py` | Evaluate one checkpoint or `--baselines` |
| `src/compare.py` | Compare completed evaluations |
| `src/report.py` | Write documents; optional `--bundle` exports a Colab ZIP |
| `src/run_tests.py` | Run pytest when requested |
| `src/status.py` | Read progress and write status |
| `src/config.py` | YAML loading and protocol definitions |
| `src/models.py` | LSTM, CNN-LSTM and contrastive loss |
| `.github/workflows/ci.yml` | Checks started manually from GitHub Actions |

Use `--help` on a script for its flags. `--model` (or `-model`) selects the
architecture. `--device cpu|cuda|auto` and `--root <path>` are overrides; other
hyperparameters belong in YAML.

Each stage script in `src/` contains its actual implementation. Shared model
definitions are in `src/models.py`, and configuration helpers in `src/config.py`.
The small execution block at the end lets other scripts reuse functions without
starting a run when they import them. There is no central dispatcher or package installation.

## 14. Optional Colab and common problems

For GPU use, prepare `python src/report.py --bundle --config configs/pilot.yaml`
locally, then follow `notebooks/colab_training.ipynb`. Its training, evaluation,
comparison and reporting cells are separate manual commands. The ZIP excludes
the attached PDF, raw data and checkpoints. Keep GPU/CPU results separate.

| Problem | Action |
| --- | --- |
| Import error | Activate the environment, install `requirements.txt`, and run `python src/<script>.py` from the project root |
| Missing metadata/waveforms | Check `data.root` and the raw folder layout |
| Missing manifest | Prepare full cache, then subset cache |
| Cache protocol mismatch | Use matching settings/cache; preprocess a new cache when changing geometry or cohort |
| Saved run already exists | Explicitly resume or choose new output paths |
| Configuration changed | Restore original settings for resume; use fresh paths for changed settings |
| No selected checkpoint | Train that model/variant first |
| No comparison results | Evaluate selected models first |
| CUDA unavailable | Use `--device cpu` |
| Another job active | Wait for the current writer before starting another |

Read the [existing assignment report](ASSIGNMENT_REPORT.md) for measured findings
and [presentation notes](PRESENTATION_NOTES.md) for the class explanation.
