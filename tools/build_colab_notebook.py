"""Build the portable Colab orchestrator; model/training code stays in src."""

import json
from pathlib import Path
from textwrap import dedent

cells = []


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": dedent(source).strip() + "\n"}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell(
    "markdown",
    """
# PTB-XL: shared-code Colab GPU study

Observe 7 seconds and forecast the next 3 seconds of each 10-second, 100-Hz ECG.
This is waveform forecasting within a recording, not clinical longitudinal prediction.
The notebook calls the project source; it does not contain a second model implementation.

Start with **pilot**. For the full workflow, change `RUN_MODE` to `full`, `EPOCHS` to
`120`, and choose a new `STUDY_ID`. Full mode runs medium screening, tuning,
the two feature-selection extensions, and four full-data 120-epoch conditions.
Set `FINAL_TEST=True` only when you intend to freeze those validation choices and
evaluate fold 10. The notebook will not use test scores to choose a recipe.
Colab runtime availability and completion time are not guaranteed. Rerun with
`RESUME=True` after interruption; checkpoints and logs persist on Drive.
""",
)
cell(
    "code",
    """
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

RUN_MODE = 'pilot'                   # pilot | medium | full
EPOCHS = 60                         # medium: 60; FULL preset: 120
BATCH_SIZE = 64
STUDY_ID = 'primary_7s3s_v3'
RESUME = False
FINAL_TEST = False
ALLOW_FULL_CPU = False
ALLOW_PRIOR_HISTORY = False          # requires a passing chronology audit
TUNE_HIDDEN_128 = False              # secondary ablation
PROJECT_SOURCE = 'drive'             # drive | upload_zip
DRIVE_PROJECT = '/content/drive/MyDrive/ptbxl_assignment'
PROJECT_DIR = '/content/ptbxl_assignment'
DATA_SOURCE = 'drive'                # drive | physionet
DATA_ROOT = '/content/drive/MyDrive/PTB-XL'
DRIVE_OUTPUT = '/content/drive/MyDrive/ptbxl_research'
SINGLE_RUN = False                   # optional manual experiment instead of the study stages
ARCHITECTURE = 'lstm'
FILTERING = 'P0'                     # P0 | P1; used by SINGLE_RUN
FEATURE_SET = 'F0'                   # F0 | F1 | F2 | expanded; used by SINGLE_RUN
VALIDATION_FIXTURE = os.environ.get('PTBXL_NOTEBOOK_FIXTURE')
LOCAL_PROJECT = os.environ.get('PTBXL_NOTEBOOK_PROJECT')
assert RUN_MODE in ('pilot', 'medium', 'full')
if RUN_MODE == 'full' and EPOCHS != 120:
    raise ValueError('The FULL preset requires EPOCHS=120. Use medium for the 60-epoch study.')
""",
)
cell(
    "markdown",
    """
## 1. Project and Drive
Use a Drive project folder containing `src/`, `configs/`, and `requirements.txt`,
or upload a source ZIP. A private GitHub repository link is not required.
ZIP extraction rejects absolute/traversal paths and links. Do not include private
papers, the dataset, environments, old caches, or unrelated files in a project ZIP.
""",
)
cell(
    "code",
    """
import sys
import shutil
import zipfile

if VALIDATION_FIXTURE:
    if not LOCAL_PROJECT:
        raise ValueError('Software validation requires PTBXL_NOTEBOOK_PROJECT.')
    project = Path(LOCAL_PROJECT).resolve()
else:
    from google.colab import drive
    drive.mount('/content/drive')
    project = Path(PROJECT_DIR)
    if PROJECT_SOURCE == 'drive':
        project = Path(DRIVE_PROJECT)
    elif PROJECT_SOURCE == 'upload_zip':
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError('Upload exactly one project ZIP.')
        name, payload = next(iter(uploaded.items()))
        import io
        project.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                clean = member.filename.replace('\\\\', '/')
                target = (project / clean).resolve()
                if not target.is_relative_to(project.resolve()) or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError('Unsafe project ZIP member: ' + member.filename)
            archive.extractall(project)
        candidates = [project] + [p.parent for p in project.glob('*/requirements.txt')]
        project = next((p for p in candidates if (p / 'src/pipeline.py').exists()), project)
    else:
        raise ValueError('Choose drive or upload_zip for PROJECT_SOURCE.')
for required in ['src/pipeline.py', 'configs/config.yaml', 'requirements.txt']:
    if not (project / required).is_file():
        raise FileNotFoundError(f'Missing {required}; check your project folder/ZIP nesting.')
os.chdir(project)
sys.path.insert(0, str(project / 'src'))
print('Project:', project)
""",
)
cell(
    "markdown",
    """
## 2. Dependencies, device, and reproducibility
Package installation retains an already compatible CUDA PyTorch build. Full mode
requires an available GPU unless the CPU override is explicit. Cross-device and
cross-version bitwise reproducibility are not guaranteed; runtime versions are recorded.
""",
)
cell(
    "code",
    """
import subprocess
if not VALIDATION_FIXTURE:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
import torch
from config import load_config, study_dir, write_json
from train import seed_everything
print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print('GPU:', properties.name, 'VRAM (GiB):', round(properties.total_memory / 2**30, 2))
if RUN_MODE == 'full' and not torch.cuda.is_available() and not ALLOW_FULL_CPU:
    raise RuntimeError('Select a Colab GPU runtime, or explicitly enable ALLOW_FULL_CPU.')
seed_everything(42)
""",
)
cell(
    "markdown",
    """
## 3. Dataset source and configuration
Drive data must contain `ptbxl_database.csv`, `scp_statements.csv`, `SHA256SUMS.txt`,
and `records100/` directly beneath `DATA_ROOT`. The download choice uses official
PhysioNet v1.0.3 metadata and only the `filename_lr` records; it never requests records500.
The waveform cache lives on `/content`, while reports/checkpoints live on Drive.
""",
)
cell(
    "code",
    """
from data import download, prepare
from fixtures import fixture_config

cfg = load_config()
if VALIDATION_FIXTURE:
    cfg = fixture_config(Path(VALIDATION_FIXTURE), cfg)
    print('SYNTHETIC SOFTWARE FIXTURE: no PTB-XL performance claims.')
else:
    cfg['study_id'] = STUDY_ID
    cfg['run_mode'] = RUN_MODE
    cfg['training']['batch_size'] = BATCH_SIZE
    cfg['training']['epochs'] = 60 if RUN_MODE == 'full' else EPOCHS
    cfg['training']['full_epochs'] = 120
    cfg['training']['allow_full_cpu'] = ALLOW_FULL_CPU
    cfg['training']['tune_hidden_128'] = TUNE_HIDDEN_128
    cfg['features']['prior_history'] = ALLOW_PRIOR_HISTORY
    cfg['evaluation']['final_test'] = False
    cfg['paths']['results'] = DRIVE_OUTPUT
    cfg['paths']['cache'] = '/content/ptbxl_cache'
    if DATA_SOURCE == 'physionet':
        cfg['data']['root'] = '/content/PTB-XL'
        download(cfg['data']['root'])
    elif DATA_SOURCE == 'drive':
        cfg['data']['root'] = DATA_ROOT
    else:
        raise ValueError('Choose drive or physionet for DATA_SOURCE.')
for required in ['ptbxl_database.csv', 'scp_statements.csv', 'SHA256SUMS.txt', 'records100']:
    if not (Path(cfg['data']['root']) / required).exists():
        raise FileNotFoundError(f'Missing dataset entry {required}; inspect DATA_ROOT nesting.')
write_json(study_dir(cfg) / 'notebook_configuration.json', cfg)
print('Effective mode:', cfg['run_mode'], 'history/target:', cfg['data']['history'], cfg['data']['horizon'])
""",
)
cell(
    "markdown",
    """
## 4. Patient folds and leakage audit
`ecg_id` identifies a waveform; `patient_id` groups repeated ECGs; `strat_fold`
defines the official patient-safe partition; `filename_lr` locates the native
100-Hz WFDB pair. Timestamps may be used only for strictly earlier same-patient
annotation history, never current or equal-time labels. Preparation verifies
checksums, units, rate, shape, and finite values and preserves exact ID manifests.
""",
)
cell(
    "code",
    """
import copy
import pandas as pd
from leakage_audit import audit
from visualize import diagnostic_figures

development_cfg = copy.deepcopy(cfg)
development_cfg['run_mode'] = 'pilot' if cfg['run_mode'] == 'pilot' else 'medium'
cache, manifest = prepare(development_cfg)
display(pd.DataFrame(manifest['splits']).T)
display(audit(development_cfg, cache))
diagnostic_figures(development_cfg, cache)
print('Fold 10 remains reserved. No test metrics were calculated.')
""",
)
cell(
    "markdown",
    """
## 5. Models, preprocessing, and tensor shapes
P0 and P1 use the same scaler fitted solely on raw training history and the same
unfiltered future target. P1 is causal; it can change phase, ST morphology, QRS
detail, and initial samples. It is not assumed to improve forecasting.
The two independent classes below share neither an encoder registry nor a training loop copy.
""",
)
cell(
    "code",
    """
from models.lstm import LSTMForecaster
from models.cnn_lstm import CNNLSTMForecaster
from preprocessing import prepare_transforms

scaler_p0 = prepare_transforms(cache, development_cfg, 'P0')
scaler_p1 = prepare_transforms(cache, development_cfg, 'P1')
assert scaler_p0 == scaler_p1
for model_class in (LSTMForecaster, CNNLSTMForecaster):
    model = model_class(static_dim=4, horizon=cfg['data']['horizon'], **cfg['model'])
    with torch.inference_mode():
        prediction = model(torch.zeros(2, cfg['data']['history'], 12), torch.zeros(2, 4))
    print(model_class.__name__, 'output:', tuple(prediction.shape),
          'active parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
""",
)
cell(
    "markdown",
    """
## 6. Pilot or controlled screening
Pilot runs two epochs and reports software pass/fail, not research rankings.
Medium screening compares model×P0/P1 with ECG-only inputs and then compares F0/F1
using validation. The optional SINGLE_RUN controls run one explicitly chosen experiment.
""",
)
cell(
    "code",
    """
from study import smoke, screen, tune, feature_extension, full
from train import train_run
from evaluate import evaluate_run

if SINGLE_RUN and not VALIDATION_FIXTURE:
    run = train_run(development_cfg, cache, ARCHITECTURE, FILTERING, FEATURE_SET, resume=RESUME)
    evaluate_run(development_cfg, cache, run)
elif cfg['run_mode'] == 'pilot':
    display(smoke(cfg, resume=RESUME))
else:
    display(screen(cfg, resume=RESUME))
""",
)
cell(
    "markdown",
    """
## 7. Learning-rate and feature-selection experiments
For medium/full workflows, tune the three predefined learning rates on validation.
Run training-only mRMR and grouped validation permutation as a separate expanded-feature
extension, keeping all twelve waveform leads. Prior diagnoses stay disabled unless
their chronology is independently audited. ARIMA remains deferred.
""",
)
cell(
    "code",
    """
if cfg['run_mode'] in ('medium', 'full') and not SINGLE_RUN:
    display(tune(cfg, resume=RESUME))
    display(feature_extension(cfg, resume=RESUME))
else:
    print('Longer screening/tuning and feature-selection runs are pending; pilot mode does not fabricate them.')
""",
)
cell(
    "markdown",
    """
## 8. FULL RUN — four matched 120-epoch conditions
This cell runs only when RUN_MODE='full'. The selected feature set and hyperparameters
are fixed within each architecture's P0/P1 pair. All official training/validation
patients are used. Epoch 60 and 120 evidence, best checkpoints, optimizer/RNG state,
and logs are saved; no early stopping shortens the requested budget.
""",
)
cell(
    "code",
    """
if cfg['run_mode'] == 'full' and not SINGLE_RUN:
    display(full(cfg, resume=RESUME))
else:
    print('Full-population 120-epoch training is pending, not a completed result.')
""",
)
cell(
    "markdown",
    """
## 9. Locked final evaluation
FINAL_TEST must be explicitly enabled. The four validation-selected checkpoint hashes,
raw scaler, configurations and cohort are frozen first. Complete fold 10 is evaluated
with a ledger: completed conditions are checked and reused; interrupted conditions
resume without changing selections. Baselines use the identical raw 7s→3s target.
""",
)
cell(
    "code",
    """
from evaluate import final_evaluation, verify_lock
from visualize import run_figures
if cfg['run_mode'] == 'full' and FINAL_TEST and not SINGLE_RUN:
    cfg['evaluation']['final_test'] = True
    final_cache, _ = prepare(cfg)
    display(final_evaluation(cfg, final_cache))
    for name in verify_lock(cfg, final_cache)['runs']:
        run_figures(cfg, final_cache, study_dir(cfg) / 'full/checkpoints' / name, 'test')
else:
    print('Final test evaluation pending or intentionally disabled. Validation remains the selection source.')
""",
)
cell(
    "markdown",
    """
## 10. Automatic evidence summary and export
The final sections appear in the required order. Missing runs remain pending.
Every numerical result is tied to its saved cohort, scaler, configuration and checkpoint.
Figures explain what is visible and what conclusion is supported; example selection
and single-run, cohort, sampling-rate, and morphology limitations remain explicit.
""",
)
cell(
    "code",
    """
from IPython.display import Markdown, Image, display
from report import generate, bundle, SUMMARY_ORDER
from config import read_json

report_path = generate(cfg)
summary = read_json(report_path.parent / 'summary.json')
if summary['warning']:
    display(Markdown(summary['warning']))
figures = report_path.parent / 'figures'
medium_figures = study_dir(cfg) / 'medium' / 'figures' / 'comparisons'
image_groups = {
    'Patient-safe folds': list(figures.glob('data/fold_counts.png')),
    'Preprocessing': list(figures.glob('data/filtered_history.png')) + list(figures.glob('data/psd.png')),
    'Training': list(figures.glob('*/learning_curves.png'))[:2],
    'Twelve-lead prediction': list(figures.glob('*/*pre_specified_12_leads.png'))[:2] + list(figures.glob('*/*median_error_12_leads.png'))[:2],
    'Evidence-based comparison': list(figures.glob('comparisons/*.png'))[:3] + (
        [p for p in medium_figures.glob('*.png') if any(k in p.name for k in ('learning_rate', 'feature_sets'))]
        if cfg['run_mode'] == 'full' else []
    ),
}
for section in SUMMARY_ORDER:
    display(Markdown('## ' + section + '\\n\\n' + summary['sections'][section]))
    for path in image_groups.get(section, []):
        display(Image(filename=str(path)))
        caption = path.with_suffix('.md')
        if caption.exists():
            display(Markdown(caption.read_text(encoding='utf-8')))
archive_path = bundle(cfg)
print('Saved report:', report_path)
print('Saved ZIP (Drive in a normal Colab run):', archive_path)
""",
)

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "cells": cells,
}
for index, item in enumerate(cells):
    item["id"] = f"ptbxl-{index:03d}"
destination = Path("notebooks/PTBXL_Full_Colab_GPU.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(destination)
