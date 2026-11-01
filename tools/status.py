"""Inspect and record progress without disturbing ongoing training."""

import argparse
from pathlib import Path

import pandas as pd

from tsa.config import VARIANTS, load_config, read_json

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pilot.yaml")
args = parser.parse_args()
cfg = load_config(args.config)

rows = []
for architecture in ("lstm", "cnn_lstm"):
    for variant in VARIANTS:
        name = f"{architecture}_{variant}"
        folder = Path(cfg["paths"]["experiments"], name)
        state = "pending"
        epochs, best, mae = 0, "-", "-"
        if (folder / "history.csv").exists():
            history = pd.read_csv(folder / "history.csv")
            epochs = len(history)
            best = f"{history.val_mse.min():.6f}"
            state = "checkpoint available"
        if (folder / "test_metrics.json").exists():
            measured = read_json(folder / "test_metrics.json")
            mae = f"{measured['mae']:.6f}"
            state = "complete"
        rows.append([name, state, str(epochs), best, mae])
lines = [
    "# Training progress",
    "",
    f"Profile: {cfg.get('profile', 'full dataset')}. Validation MSE is standardized; test MAE is in original mV.",
    "Only completed models for this cohort have test metrics.",
    "",
    "| Run | State | Epochs | Best validation MSE | Test MAE (mV) |",
    "| --- | --- | --- | --- | --- |",
]
lines += ["| " + " | ".join(row) + " |" for row in rows]
lines += [
    "",
    "Resume remaining work with:",
    "",
    "```powershell",
    rf".\.venv\Scripts\python -u -m tsa run-experiments --config {args.config}",
    "```",
    "",
    "Rerunning this command resumes last completed epochs and skips measured completed runs.",
    f"Regenerate with `python tools/status.py --config {args.config}`.",
]
Path(cfg["paths"]["experiments"]).mkdir(parents=True, exist_ok=True)
Path(cfg["paths"]["experiments"], "TRAINING_STATUS.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
print("\n".join(lines))
