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
of `records100` only. The notebook starts in pilot mode. Its FULL preset runs
the staged study, saves checkpoints and logs to Drive, and exports
`PTBXL_Research_Results.zip`.

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
