# PTB-XL waveform forecasting

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

## Colab

Open `notebooks/PTBXL_Full_Colab_GPU.ipynb`. Supply a Drive project folder or
uploaded project ZIP, then choose a Drive dataset or official PhysioNet download
of records100 only. No private GitHub access or Windows paths are required.
The notebook starts in pilot mode. Its FULL preset runs the staged study,
persists checkpoints/logs to Drive and exports `PTBXL_Research_Results.zip`.
GPU availability and completion are not guaranteed; pending runs stay pending.

## Code and explanations

- `src/data.py`: checksum/quality checks, patient selection, caches and DataLoaders.
- `src/preprocessing.py`: causal history filtering and training-fitted transforms.
- `src/models/`: independent LSTM and CNN-LSTM classes with early conditioning.
- `src/train.py`: the shared training/validation loop and resumable checkpoints.
- `src/evaluate.py`: explicit inference, baselines and locked final testing.
- `src/feature_selection.py`: the separate expanded-feature extension.
- `src/visualize.py`, `src/compare.py`, `src/report.py`: figures and saved-evidence reporting.
- `src/pipeline.py`: the small shared CLI; the notebook calls the same functions.

Read `docs/METHODOLOGY.md`, `docs/EQUATIONS.md`, `docs/DATA_DICTIONARY.md`, and
`docs/DATA_LEAKAGE_AUDIT.md`. Generated `docs/RESULTS.md` reflects actual artifacts.
See `docs/EXECUTION_STATUS.md` for what ran locally and what remains pending.
Equations, units, patient weighting, filter limitations and null correlations are
explained explicitly. The final summary follows the required nine-section order.

The previous implementation is preserved in `archive/legacy_2ad3e68`. Existing
weights and measured outputs remain untouched; old horizons are not new results.
See `docs/CHANGE_MAP.md` for what was kept, fixed, simplified and archived.
