"""Rebuild lightweight notebooks; no data, credentials or reference PDF is embedded."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def save(name, cells):
    # Default to the active small-data profile; use base.yaml for later full-data runs.
    for cell in cells:
        source = "".join(cell["source"])
        source = source.replace("configs/base.yaml", "configs/pilot.yaml")
        source = source.replace("data/preprocessed", "data/pilot")
        source = source.replace("outputs/", "outputs/pilot/")
        source = source.replace(
            "drive_dir / 'outputs/pilot/model_comparison.csv'",
            "drive_dir / 'pilot/outputs/model_comparison.csv'",
        )
        source = source.replace("str(drive_dir / key)", "str(drive_dir / 'pilot' / key)")
        cell["source"] = source.splitlines(keepends=True)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i}"
    Path("notebooks").mkdir(exist_ok=True)
    Path("notebooks", name).write_text(json.dumps(notebook, indent=2), encoding="utf-8")


save(
    "assignment_walkthrough.ipynb",
    [
        markdown(
            "# Patient-conditioned ECG forecasting\n\nRun the CLI workflow in README first. "
            "Select the project virtual-environment kernel. This notebook reads measured "
            "results; it does not invent metrics or retrain models when opened."
        ),
        code(
            "from pathlib import Path\nimport os\nif not Path('src/tsa').exists() and Path('../src/tsa').exists():\n"
            "    os.chdir('..')\nfrom tsa.config import load_config, read_json\n"
            "cfg = load_config('configs/base.yaml')\nmanifest = read_json('data/preprocessed/manifest.json')\n"
            "manifest['protocol']"
        ),
        markdown(
            "## Patient splits and conditioning\n\nAge and sex are fixed covariates. "
            "NORM/MI/STTC/CD/HYP history comes only from strictly earlier ECGs. "
            "Current-record diagnoses below are used only for stratified evaluation."
        ),
        code(
            "import pandas as pd\nmeta = {s: pd.read_csv(f'data/preprocessed/metadata_{s}.csv') "
            "for s in ('train','val','test')}\npd.DataFrame({s: {'records': len(m), "
            "'patients': m.patient_id.nunique(), 'with_history': int(m.history_available.sum())} "
            "for s,m in meta.items()}).T"
        ),
        markdown(
            "## Model input/output shapes\n\n500 Hz, 300 past samples (0.6 s), "
            "150 future samples (0.3 s), all 12 leads. Contrastive projection receives "
            "observed dynamics before static fusion."
        ),
        code(
            "from tsa.data import ECGDataset\nfrom tsa.models import build_model\n"
            "ds = ECGDataset(cfg, 'train', 'patient_history')\nbatch = ds[0]\n"
            "model = build_model(cfg, 'cnn_lstm', 10).eval()\nimport torch\n"
            "with torch.inference_mode():\n    forecast = model(batch['x'][None], batch['static'][None])\n"
            "print(batch['x'].shape, batch['static'].shape, forecast.shape)"
        ),
        code("pd.read_csv('outputs/model_comparison.csv')"),
        code("pd.read_csv('outputs/paired_patient_bootstrap.csv')"),
        code(
            "from IPython.display import Image, display\n"
            "display(Image('outputs/model_comparison.png'))\n"
            "display(Image('outputs/cnn_lstm_patient_history/learning_curves.png'))\n"
            "display(Image('outputs/cnn_lstm_patient_history/forecast_lowest_id.png'))"
        ),
        markdown(
            "## Discuss\n\nCompare demographic versus ECG-only, history versus demographics, "
            "and contrastive versus history-only runs. Negative paired B-A intervals favor B. "
            "Compare architectures under each condition. Inspect sharp peaks and timing "
            "as well as mean error. Read `Project_Documents/ASSIGNMENT_REPORT.md` "
            "and `Project_Documents/PRESENTATION_NOTES.md` for equations and literature caveats."
        ),
    ],
)

save(
    "colab_training.ipynb",
    [
        markdown(
            "# Colab GPU: same patient-conditioned experiment\n\nFirst run local audit/preprocess, "
            "then `python -m tsa bundle`. Manually put ONLY `outputs/colab_bundle.zip` in "
            "`MyDrive/ptbxl_assignment/`. The bundle excludes the reference PDF, raw dataset, "
            "and checkpoints. Select **Runtime > Change runtime type > T4 GPU**. "
            "This notebook does not upload files automatically."
        ),
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        code(
            "from pathlib import Path\nfrom zipfile import ZipFile\nimport os\n"
            "drive_dir = Path('/content/drive/MyDrive/ptbxl_assignment')\n"
            "project = Path('/content/ptbxl_assignment')\nproject.mkdir(exist_ok=True)\n"
            "with ZipFile(drive_dir / 'colab_bundle.zip') as archive:\n"
            "    assert all(not n.lower().endswith('.pdf') and '..' not in Path(n).parts "
            "and not Path(n).is_absolute() for n in archive.namelist())\n"
            "    archive.extractall(project)\nos.chdir(project)"
        ),
        code(
            "%pip install -q -e .\nimport torch\nprint(torch.__version__, 'CUDA:', torch.cuda.is_available())\n"
            "assert torch.cuda.is_available(), 'Select a GPU runtime before full training.'"
        ),
        markdown(
            "## Persist outputs and checkpoints\n\nKeep arrays on Colab local disk for speed, "
            "but save artifacts to Drive each epoch. Re-running the same configuration "
            "resumes checkpoints or skips completed runs. CPU/GPU changes can affect "
            "numerical behavior; exact cross-device reproducibility is not guaranteed."
        ),
        code(
            "import yaml\nfrom tsa.config import load_config\ncfg = load_config('configs/base.yaml')\n"
            "cfg['data']['root'] = 'PTB-XL'  # raw files not needed for a validated cache\n"
            "cfg['data']['cache'] = 'data/preprocessed'\n"
            "cfg['training']['device'] = 'auto'\n"
            "for key in ('experiments','outputs','documents'):\n"
            "    cfg['paths'][key] = str(drive_dir / key)\n"
            "Path('configs/colab.yaml').write_text(yaml.safe_dump(cfg))\n"
            "from tsa.data import validate_cache\nvalidate_cache(cfg)['fingerprint']"
        ),
        code("!python -u -m tsa benchmark --config configs/colab.yaml"),
        code("!python -u -m tsa run-experiments --config configs/colab.yaml"),
        code("import pandas as pd\npd.read_csv(drive_dir / 'outputs/model_comparison.csv')"),
        markdown(
            "## Return results to the local project\n\nDownload `experiments`, `outputs`, "
            "and `documents` from this dedicated Drive folder. Place documents under local "
            "`Project_Documents`, retain experiment names, then run "
            "`python -m tsa report --config configs/base.yaml`. Do not mix fingerprints. "
            "Keep existing CPU results separately if retaining both CPU and GPU results."
        ),
    ],
)
