"""Execute the Week 5 notebook against an isolated synthetic fixture."""

import argparse
import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    from fixtures import synthetic_dataset

    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    spec_root = args.work / "jupyter"
    spec = spec_root / "kernels" / "ptbxl-validation"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "PTB-XL isolated validation",
                "language": "python",
            }
        )
    )
    env = os.environ.copy()
    for name, relative in (
        ("JUPYTER_CONFIG_DIR", "config"),
        ("JUPYTER_DATA_DIR", "data"),
        ("JUPYTER_RUNTIME_DIR", "runtime"),
        ("IPYTHONDIR", "ipython"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        directory = args.work / relative
        directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
        os.environ[name] = str(directory)
    env["JUPYTER_PATH"] = str(spec_root)
    os.environ["JUPYTER_PATH"] = str(spec_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    fixture = synthetic_dataset(args.work / "fixture")
    notebook = nbformat.read(project / "notebooks/Week5_ECG_Forecasting.ipynb", as_version=4)
    settings = {
        "project_root": str(project),
        "data_root": str(fixture),
        "cache_dir": str(args.work / "cache"),
        "results_dir": str(args.work / "results"),
        "study_id": "week5_synthetic_validation",
        "mode": "full",
        "smoke_test": False,
        "epochs": 2,
        "allow_full_cpu": True,
        "device": "cpu",
        "hidden_size": 8,
        "num_layers": 1,
        "batch_size": 4,
        "run_training": True,
        "run_all_experiments": True,
        "use_static_features": True,
        "run_training_ablation": True,
        "smaller_train_patients": 2,
        "shorter_epochs": 1,
        "final_test": True,
    }
    notebook.cells[1].source += f"\nCONFIG.update({settings!r})"
    env_cell = next(cell for cell in notebook.cells if "cfg = configure(CONFIG)" in cell.source)
    env_cell.source += '\ncfg["data"]["synthetic"] = True\nprint("SYNTHETIC SOFTWARE TEST ONLY")'
    nbformat.validate(notebook)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="ptbxl-validation",
        resources={"metadata": {"path": str(args.work)}},
        allow_errors=False,
    )
    client.execute(env=env)
    destination = args.work / "Week5_ECG_Forecasting_executed_synthetic.ipynb"
    ledger = args.work / "results/week5_synthetic_validation/full/test_ledger.json"
    final = json.loads(ledger.read_text(encoding="utf-8"))
    if len(final["conditions"]) != 6 or any(
        entry["state"] != "completed" for entry in final["conditions"].values()
    ):
        raise RuntimeError("Expected four frozen fixture models and two baseline evaluations.")
    nbformat.write(notebook, destination)
    completed = sum(
        cell.cell_type == "code" and cell.execution_count is not None for cell in notebook.cells
    )
    result = {
        "passed": True,
        "executed_code_cells": completed,
        "fixture": "synthetic software validation only",
        "gpu_training_executed": False,
        "environment": sys.executable,
        "executed_notebook": str(destination),
    }
    (args.work / "notebook_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
