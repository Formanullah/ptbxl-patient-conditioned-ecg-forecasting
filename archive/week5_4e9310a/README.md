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

## Notebook: local or Colab

Open [Week5_ECG_Forecasting.ipynb](notebooks/Week5_ECG_Forecasting.ipynb) in Jupyter,
VS Code, or Colab. It demonstrates the complete workflow and imports shared
implementations from `src/`. The earlier standalone notebook is preserved in
[the notebook archive](archive/previous_notebooks/).

- **Local:** use your project Python kernel, open the notebook from this
  repository, then set `CONFIG['data_root']` if PTB-XL is stored elsewhere.
- **Colab:** put this project on Drive or clone it into the runtime. Set
  `CONFIG['project_root']` to the folder containing `src/`, `configs/`, and
  `requirements.txt`. Set the data folder separately; for Drive it defaults to
  `My Drive/Colab Notebooks/PTB-XL`. With `data_source='physionet'`, use a runtime
  directory for `data_root`; only selected train/validation waveform pairs are
  downloaded before training. Keep `results_dir` on Drive for long runs.

The first cell configures the run. It is set to train E1–E4 for 120 epochs
each on all training and validation patients, with results under
`results/week5_full/full/` locally. Select a CUDA kernel or a Colab GPU for this
configuration. For a two-epoch pilot, set `mode='pilot'` and `smoke_test=True`.
Set `run_training=False` to inspect saved evidence, or set
`run_all_experiments=False` and choose one `experiment_id` to train a single
condition. Use a new `study_id` when the cohort or settings change. Compatible
finished runs are reused. After all four full-data runs finish, set
`final_test=True` to evaluate fold 10.

## Results and verification

The [Week 5 results](docs/WEEK5_RESULTS.md) summarize the matched real-data CPU
validation runs and their limitations. The earlier
[CPU pilot report](docs/RESULTS.md) describes the previous protocol and remains
separate. Use `python tools/validate_notebook.py --work <scratch-folder>` to
execute the Week 5 notebook end to end on a synthetic software fixture, including
the guarded final-test path. Fixture metrics are not PTB-XL evidence.

The 120-epoch full-data GPU runs and real fold-10 scoring are pending.
The short CPU pilot measures only the stated cohort and epoch count.

The Week 4 CLI entry point is `src/pipeline.py`. mRMR and grouped permutation
importance remain an earlier, separate extension; they are not part of the
Week 5 notebook. Old checkpoints retain their saved source provenance.
See [methodology](docs/METHODOLOGY.md), [equations](docs/EQUATIONS.md), and the
[data dictionary](docs/DATA_DICTIONARY.md). The [historical archive](archive/legacy_2ad3e68/README_ARCHIVE.md)
explains earlier protocols and which old reports lack supporting artifacts.
