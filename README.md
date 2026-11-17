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

`configs/config.yaml` is the default CLI configuration and selects study
`primary_7s3s_v3` in pilot mode. Pilot mode defaults to 300
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
or the official PhysioNet download of `records100` only. The notebook currently
defaults to `RUN_MODE='full'`, `EPOCHS=120`, `STUDY_ID='primary_7s3s_v5'`,
`ALLOW_FULL_CPU=True`, and `FINAL_TEST=False`. For a quick software check, set
`RUN_MODE='pilot'`; this uses the two-epoch pilot budget. The first cell controls
patient counts for pilot and medium screening; use a new `STUDY_ID` when changing
the cohort. Section 4
shows the resulting ECG and patient counts, including reserved test metadata.
Full mode uses the official full cohorts and requires CUDA unless explicitly
enabled on CPU. Results are exported as `PTBXL_Research_Results.zip`.

## Results and verification

The [research report](docs/RESULTS.md) now covers the completed full-population
study `primary_7s3s_v5`. Saved artifacts record real-data screening, learning-rate
tuning, and both feature-selection extensions, with 16 completed 60-epoch medium
runs. The medium cohort contains 1,731 ECGs from 1,500 training patients and
2,183 ECGs from all 1,942 official validation patients.

All four full-population conditions completed 120 epochs on **CPU**, with
`allow_full_cpu` explicitly enabled. Training used 17,418 ECGs from 15,023
patients; validation used 2,183 ECGs from 1,942 patients. All four selected
recipes use F0 (ECG-only inputs). Validation selects the best checkpoint within
each completed run, so its selected epoch can precede epoch 120.

| Condition | Learning rate | Selected epoch | Validation patient-macro MAE (mV) | Validation RMSE (mV) |
| --- | --- | --- | --- | --- |
| CNN-LSTM, raw history (P0), F0 | 0.001 | 116 | 0.105195 | 0.226551 |
| CNN-LSTM, causal filter (P1), F0 | 0.001 | 43 | 0.108725 | 0.227847 |
| LSTM, raw history (P0), F0 | 0.0005 | 29 | 0.111099 | 0.234362 |
| LSTM, causal filter (P1), F0 | 0.0005 | 99 | 0.110185 | 0.231235 |
| Persistence baseline | — | — | 0.160367 | 0.314331 |
| Repeat-window baseline | — | — | 0.170786 | 0.326306 |

Raw-input CNN-LSTM has the lowest validation patient-macro MAE, approximately
34.4% below persistence. Filtering increased CNN-LSTM error by approximately
3.36% and reduced LSTM error by approximately 0.82%. These are validation
results from one training seed; they do not establish clinical usefulness.
Pilot and synthetic scores remain excluded from research rankings.

The four-condition selection is frozen in
`results/primary_7s3s_v5/full/selection.json`. Final fold-10 scoring is pending:
there is no completed test ledger, and the reserved 2,198 ECGs from 1,904
patients have not been prepared for this study. Enabling final scoring requires
the saved study configuration and frozen artifacts.

The default CLI configuration still targets the earlier pilot study. To report
or export the completed study from this local checkout, use its saved configuration:

```sh
python src/pipeline.py report --config results/primary_7s3s_v5/notebook_configuration.json --mode full
python src/pipeline.py export --config results/primary_7s3s_v5/notebook_configuration.json --mode full
```

`results/`, dataset caches, and model weights are excluded from Git; the commands
above require the local artifacts. The tracked report in `docs/RESULTS.md`
contains the recorded results for readers without those files.

Verification on 17 November 2026 with `python src/run_tests.py -q` returned
**27 passed, 1 failed**. The failing test,
`test_notebook_schema_plain_python_and_shared_code`, expects the notebook to
default to `RUN_MODE = 'pilot'`; its current setting is `full`. The remaining
tests passed, including patient isolation, causal preprocessing, training-only
transforms, checkpoint/RNG recovery, metrics, and final-test locking. All four
full-run source fingerprints match the current `src/` files, and their saved
evaluation receipts and checkpoint provenance passed the report collector's checks.

The shared CLI is `src/pipeline.py`; the notebook calls the same experiment code.
See [methodology](docs/METHODOLOGY.md), [equations](docs/EQUATIONS.md), and the
[data dictionary](docs/DATA_DICTIONARY.md). The [historical archive](archive/legacy_2ad3e68/README_ARCHIVE.md)
explains earlier protocols and which old reports lack supporting artifacts.
