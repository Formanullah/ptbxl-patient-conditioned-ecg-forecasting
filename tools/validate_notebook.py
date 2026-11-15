"""Execute the unchanged notebook in a fresh kernel with an isolated synthetic dataset."""

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
    env["PTBXL_NOTEBOOK_FIXTURE"] = str(args.work / "fixture")
    env["PTBXL_NOTEBOOK_PROJECT"] = str(project)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    notebook = nbformat.read(project / "notebooks/PTBXL_Full_Colab_GPU.ipynb", as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="ptbxl-validation",
        resources={"metadata": {"path": str(args.work)}},
        allow_errors=False,
    )
    client.execute(env=env)
    destination = args.work / "PTBXL_Full_Colab_GPU_executed_synthetic.ipynb"
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
