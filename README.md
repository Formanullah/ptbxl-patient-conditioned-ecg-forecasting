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

## CNN-LSTM and TFT comparison

For a matched validation run, use the same schema-3 configuration and cohort
for both models. Start with raw ECG-only history (P0/F0).

```sh
python src/pipeline.py train --mode medium --model cnn_lstm --preprocessing P0 --features F0
python src/pipeline.py train --mode medium --model tft --preprocessing P0 --features F0
python src/pipeline.py evaluate --mode medium --run cnn_lstm_P0_F0_lr0.0005_h64_e60 tft_P0_F0_lr0.0005_h64_e60
python src/pipeline.py report --mode medium
```

The example run names match `configs/config.yaml`. If you change the
configuration, use the run names printed by training. The report includes
saved validation metrics in mV, including patient-macro MAE. For a quick
software check, use pilot mode and pass the same
`--fixture-dir .cache/synthetic-pair` to each command.

The CLI's `tft` option uses the existing TFT-style encoder in
`src/models/tft.py`. The PyTorch Forecasting notebook in
`notebooks/TFT.ipynb` has its own five-second history protocol and
checkpoint format.

## Original recurrent study

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

The published pilot report describes that specific run. To verify a later
experiment, use its saved configuration, checkpoint and validation receipts.
Pilot and synthetic scores do not rank the research conditions.

The shared CLI is `src/pipeline.py`; the notebook calls the same experiment code.
See [methodology](docs/METHODOLOGY.md), [equations](docs/EQUATIONS.md), and the
[data dictionary](docs/DATA_DICTIONARY.md). The [historical archive](archive/legacy_2ad3e68/README_ARCHIVE.md)
explains earlier protocols and which old reports lack supporting artifacts.
