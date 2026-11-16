# Configuration and run guide

Use this guide in order. Every command selects one stage/model. The main configuration is an expanded development candidate; its accuracy requires retraining. The previous 1,500/300/100 run passed 60 tests and its real-cache leakage audit, and its measured report is archived with that cohort's outputs.

## 1. Set up the environment

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.venv` only once. On later days, activate it and continue. Do not create `pytest.py`: `python src/run_tests.py -q` is the existing test runner. `-q` means quieter test output.

## 2. Choose data in one file

Edit `configs/training.yaml`:

```yaml
data:
  root: PTB-XL
  sampling_rate: 100
  history_seconds: 5.0
  horizon_seconds: 3.0
  window_stride_seconds: 1.0
subset:
  mode: custom
  seed: 42
  patients: {train: 4500, val: 600, test: 100}
```

- `root`: the folder containing metadata and native `records100/` waveforms.
- `history_seconds`: the ECG the model observes. Five seconds at 100 Hz gives 500 samples.
- `horizon_seconds`: the future ECG the model predicts. Three seconds gives 300 samples.
- `window_stride_seconds`: spacing between training-window starts. Three valid 5-to-3-second windows fit in a ten-second recording with a one-second stride.
- `patients`: whole patients selected inside each official split, retaining their recordings.
- `mode: full`: use every official patient; the patient-count settings are then ignored.

For fractions instead of counts, replace `patients` with `fractions: {train: 0.2, val: 0.2, test: 0.1}` while keeping `mode: custom`. Values must lie in (0,1]. Do not leave both counts and fractions: fractions take precedence.

The loader converts seconds to samples. You do not edit `history` or `horizon` sample counts separately. The complete observed+future window must fit within ten seconds. Training statistics come only from training inputs.

Dataset settings determine a cohort identifier and separate locations under `data/`, `experiments/` and `outputs/`. The same cohort can share models while filtered/scaled inputs have separate caches. Explicit paths in a saved checkpoint configuration are preserved for resume. Never edit those snapshot paths to disguise a different cohort.

## 3. Check and prepare

```powershell
python src/run_tests.py -q
python src/preprocessing.py
python src/leakage_audit.py
```

`preprocessing.py` reads selected raw ECGs, verifies waveforms, constructs windows and strictly prior-history features, fits training-only transformations and saves memory-mapped arrays plus a manifest. It reuses an existing matching cache and rejects an incompatible one.

`leakage_audit.py` checks patient/record/window separation, train-only statistics and future-mutation invariance. This is required again for a changed cohort. An optional complete raw-waveform audit is `python src/preprocessing.py --audit`; it scans the raw metadata population, not only the selected subset.

The previous 1,500/300/100 cohort contained 1,731 / 336 / 114 recordings and 5,193 / 336 / 114 windows. Its tests, leakage audit, two 60-epoch model runs and validation/baseline evaluations completed successfully. `configs/reference_1500.yaml` preserves those settings. The expanded cohort receives separate cache/result paths; use its own manifest for measured counts.

## 4. Train one model

```powershell
python src/train.py --model lstm --variant patient_history
```

This trains only that model. Training reads training/validation loaders, saves epoch metrics, selects `best.pt` using validation patient-macro MAE in mV, and saves resumable `last.pt`. It does not run the final test evaluation or automatically train other models.

Updated candidate settings: hidden size 128, depth 2, dropout 0.2, static embedding 16, head width 128; AdamW learning rate 0.0005 and weight decay 0.0005; batch size 64. The epoch cap stays 60, with early stopping after 10 non-improving epochs. The plateau scheduler halves the learning rate after six non-improving validation epochs. This provides a lower-rate training opportunity before early stopping. Checkpoint selection remains validation patient-macro MAE in mV. Increased capacity and patient counts do not guarantee heartbeat preservation.

Choose `--model lstm` or `--model cnn_lstm` and one variant:

| Variant | Static inputs |
| --- | --- |
| `ecg_only` | None |
| `demographics` | Age, sex and missingness flags: 4 values |
| `patient_history` | Demographics, prior diagnostic superclasses and history flag: 10 values |
| `patient_history_contrastive` | The same 10 inputs, plus contrastive training |
| `expanded` | Also height, weight, BMI and observed HR/RR features: 22 values |
| `mrmr` / `permutation` | Fingerprint-checked selected feature groups |

For CPU/GPU selection, use `--device cpu` or `--device cuda`, or leave `training.device: auto`. GPU availability is not guaranteed by the configuration.

## 5. Evaluate, then compare

```powershell
python src/evaluate.py --model lstm --variant patient_history --split val
python src/train.py --model cnn_lstm --variant patient_history
python src/evaluate.py --model cnn_lstm --variant patient_history --split val
python src/evaluate.py --baselines --split val
python src/report.py
```

Evaluate explicitly after each desired run. The baseline command evaluates last-value persistence, repeated observed samples and a constant training-median reference. It does not train a neural model.

Compare architectures with the same features and training settings. Compare ECG-only → demographics → patient history to study conditioning. Compare patient history → contrastive history for the auxiliary loss. Compare early/late fusion while keeping the other settings fixed. Low voltage MAE does not necessarily mean realistic R-peaks; inspect the waveform and QRS plots.

`report.py` refreshes the single active `Project_Documents/ASSIGNMENT_REPORT.md`, including the current requested/actual cohort, execution state, available metrics, settings, uncertainty and figures. It does not retrain or generate predictions. Before training it reports pending results. Different cohort/window identifiers and provenance checks prevent old scores from being reused as larger-dataset scores.

Run `python src/compare.py` if you need comparison CSVs/charts without rewriting the report. No automatic watcher or run-everything command exists.

## 6. Change parameters without overwriting a run

Settings belong to these sections:

| Section | Examples |
| --- | --- |
| `data` / `subset` | Sampling rate, windows, stride, patient counts |
| `preprocessing` | `filter: none/highpass/bandpass`; `scaling: zscore/robust` |
| `model` | Hidden size, depth, dropout, static embedding, early/late fusion, kernels, pooling |
| `training` | Epochs, learning rate, batch size, optimizer weight decay, loss, scheduler |
| `contrastive` | Weight, temperature, jitter, gain range |
| `evaluation` | Bootstrap count, peak tolerance, final-test permission |

Use the optional `configs/experiments.yaml` for named changes. For example, `lstm_hidden_64` changes hidden size and `lstm_lr_0001` changes learning rate. These inherit the main data configuration:

```powershell
python src/train.py --config configs/experiments.yaml --experiment lstm_hidden_64
python src/evaluate.py --config configs/experiments.yaml --experiment lstm_hidden_64 --split val
python src/report.py
```

To add another experiment, give it a unique name:

```yaml
  lstm_lr_0001_100epochs:
    training: {lr: 0.0001, epochs: 100}
```

Put it under `experiments:`. Changing multiple factors is allowed, but it no longer isolates one factor's effect. Do not edit a completed run's saved configuration. For full 60/100-epoch comparisons use distinct names and `early_stopping: false`.

The expanded recipe changes multiple model/training factors together. `cnn_lstm_previous_settings` supplies the previous complete recipe on the same expanded cohort. `cnn_lstm_1500_patients` tests the updated recipe with fewer training patients while retaining the same 600 validation patients; prepare that recipe's separate cache first. For a morphology experiment, `cnn_lstm_morphology` adds derivative loss weight 0.1. Compare it to the updated CNN-LSTM using waveform MAE/RMSE, QRS scores and overlays. Do not assume a derivative penalty solves the flat-forecast failure. Old 300-patient and new 600-patient validation means describe different populations.

For a different filter/scaler, prepare that recipe's cache before training:

```powershell
python src/preprocessing.py --config configs/experiments.yaml --experiment lstm_highpass
python src/train.py --config configs/experiments.yaml --experiment lstm_highpass
python src/evaluate.py --config configs/experiments.yaml --experiment lstm_highpass --split val
python src/preprocessing_diagnostics.py --config configs/experiments.yaml --experiment lstm_highpass
```

## 7. Feature selection, if needed

Train an expanded-feature model before grouped permutation importance:

```powershell
python src/train.py --config configs/experiments.yaml --experiment lstm_expanded
python src/feature_selection.py --config configs/experiments.yaml --experiment lstm_expanded --method mrmr --count 3
python src/feature_selection.py --config configs/experiments.yaml --experiment lstm_expanded --method permutation --count 3 --repeats 5
python src/train.py --config configs/experiments.yaml --experiment lstm_mrmr
python src/evaluate.py --config configs/experiments.yaml --experiment lstm_mrmr --split val
```

mRMR uses training features and training waveform-summary targets. Grouped permutation ranks a training-fitted model using validation errors, grouping flags with variables. `--count 3` means at most three groups, not necessarily three individual columns. Run the corresponding permutation-selected model in the same manner. Artifacts from another cohort are rejected.

## 8. Resume and locate artifacts

A run directory contains `config.json`, `feature_manifest.json`, `history.csv`, `training_metrics.json`, `best.pt` and `last.pt`. Evaluation adds `val_metrics.json`, per-record metrics and predictions. Output figures live under `outputs/<cohort>/<run>/val/`; learning curves are under `outputs/<cohort>/<run>/`.

Resume the exact recorded settings:

```powershell
python src/train.py --config experiments/COHORT/RUN/config.json --resume
```

If the saved run already reached early-stopping patience but you intentionally want every configured epoch, add `--finish-budget`. This retains the checkpoint's model, optimizer, scheduler, data and scientific configuration while bypassing only the stop condition:

```powershell
python src/train.py --config experiments/COHORT/RUN/config.json --resume --finish-budget
```

Replace `COHORT/RUN` with the actual folder shown by your configuration/output paths. Without `--resume`, existing checkpoints cause an explicit error. Resume restores optimizer, scheduler and random-generator state. A partial interrupted epoch is repeated.

## 9. Increase the dataset and refresh the report

1. Change counts, fractions or `subset.mode` in training.yaml.
2. Run preprocessing and the leakage audit for the new cohort.
3. Train each selected model manually.
4. Evaluate on validation and run baselines.
5. Run `python src/report.py` again.

The active report is replaced with the selected cohort's findings. It is not a live document and will not update merely because YAML was edited. Old experiment folders remain separate.

## 10. Final official test evaluation

Only after full-population validation selection:

1. Set `subset.mode: full`, prepare, audit, train and validate the desired candidates.
2. Freeze chosen full-cohort run names, for example:

```powershell
python src/evaluate.py --freeze lstm_patient_history cnn_lstm_patient_history
```

3. Set `evaluation.allow_test: true` in the main configuration. Its selection path defaults to that full cohort's `selection.json`.
4. Evaluate only frozen candidates, then report the test measurements:

```powershell
python src/evaluate.py --model lstm --variant patient_history --split test
python src/evaluate.py --model cnn_lstm --variant patient_history --split test
python src/evaluate.py --baselines --split test
python src/report.py --split test
```

The example names are possible selections, not a claim those full-data models already exist. Fold-10 inference requires full mode, explicit permission, matching provenance and frozen checkpoint hashes. Different filter/scaler fingerprints require separate selection records. Do not use test outcomes to choose parameters.

## Script responsibilities and architecture

| File | Responsibility |
| --- | --- |
| `config.py` | YAML inheritance, validation, derived paths and protocol fingerprints |
| `preprocessing.py` | Raw loading/audit, causal transforms, windows, static features, cache and DataLoaders |
| `models.py` | LSTM/CNN-LSTM, fusion, augmentation and contrastive objective |
| `train.py` | Optimization, validation checkpoint selection, epoch logging and resume |
| `metrics.py` | Pure numerical waveform metrics, inverse scaling and patient bootstrap |
| `morphology.py` | Detector-derived peak matching and QRS metrics |
| `evaluate.py` | Explicit inference, baselines, figures and final selection freeze |
| `compare.py` | Comparable saved runs, pending states, tables and confidence intervals |
| `report.py` | Configuration-driven assignment report |
| `feature_selection.py` | Optional mRMR and grouped permutation |
| `leakage_audit.py` | Real-cache leakage checks |
| `preprocessing_diagnostics.py` | Optional raw/filtered spectra and waveform plots |
| `run_tests.py` | Existing pytest runner |

All scripts stay in `src/`. Numerical metrics do not import training or plotting. Reporting reads saved artifacts and never invokes training/inference. There is one implementation for preprocessing and one for evaluation; earlier experiment measurements remain as historical outputs.

## Cleanup awaiting explicit approval

Automatic approval review rejected removing the following retired files, citing the risk that deleting source/configuration/document files could break references. They remain in place. The active workflow above no longer imports the retired modules or uses the earlier presets. These are proposed removals, not completed removals:

```text
src/research.py
src/preparation.py
src/status.py
configs/base.yaml
configs/final_full.yaml
configs/finalists.yaml
configs/legacy_500hz.yaml
configs/local.yaml
configs/medium.yaml
configs/pilot.yaml
configs/research.yaml
configs/selected_study.yaml
configs/study.yaml
Project_Documents/IMPLEMENTATION_PLAN.md
Project_Documents/research_100hz/RESEARCH_REPORT.md
Project_Documents/research_100hz/PRESENTATION_NOTES.md
notebooks/assignment_walkthrough.ipynb
notebooks/colab_training.ipynb
requirements-tested.txt
```

The current preprocessing/evaluation/comparison/report implementations and tests were updated to use the two active configs. Legacy training-selection compatibility remains. Reversible source/document/notebook copies exist in `.cache/before_cleanup.zip`; historical reports, the latest legacy presentation, configuration text and tested dependency versions were also copied to the existing historical result folders. Raw data, caches, experiments and measured CSV/PNG artifacts are not proposed for removal.
