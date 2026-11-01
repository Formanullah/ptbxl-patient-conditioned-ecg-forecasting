"""Compare evaluated models and patient-bootstrap confidence intervals."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config, read_json
from evaluate import save_figure


def bootstrap_pair(a, b, samples=1000, seed=42):
    """Paired patient-macro MAE delta B minus A; negative favors B."""
    if set(a.ecg_id) != set(b.ecg_id):
        raise ValueError("Cannot pair different test record sets")
    joined = a[["ecg_id", "patient_id", "mae_mV"]].merge(
        b[["ecg_id", "patient_id", "mae_mV"]],
        on="ecg_id",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if not joined.patient_id_a.equals(joined.patient_id_b):
        raise ValueError("Patient assignments differ between runs")
    joined["delta"] = joined.mae_mV_b - joined.mae_mV_a
    delta = joined.groupby("patient_id_a").delta.mean().to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.empty(samples)
    for i in range(samples):
        boots[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    return {
        "delta_patient_macro_mae_mV": float(delta.mean()),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "patients": len(delta),
        "bootstrap_samples": samples,
    }


def compare_runs(cfg):
    import matplotlib.pyplot as plt

    root = Path(cfg["paths"]["experiments"])
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    rows, fingerprints = [], set()
    for run in sorted(root.iterdir()):
        if not (run / "test_metrics.json").exists():
            continue
        metrics, config = read_json(run / "test_metrics.json"), read_json(run / "config.json")
        if config.get("smoke"):
            continue
        fingerprints.add(metrics["fingerprint"])
        rows.append(
            {
                "run": run.name,
                **{
                    k: metrics[k]
                    for k in (
                        "mae",
                        "rmse",
                        "mse",
                        "r2",
                        "pearson_mean",
                        "patient_macro_mae",
                        "n_test_records",
                        "n_test_patients",
                    )
                },
                "epochs": metrics.get("epochs_completed", 0),
                "best_epoch": metrics.get("best_epoch"),
                "training_seconds": metrics.get("training_seconds", 0),
                "parameters": metrics.get("parameters", 0),
            }
        )
    if len(fingerprints) > 1:
        raise ValueError("Refusing comparison of mismatched protocols/caches")
    if not rows:
        raise ValueError("No measured full-data results found")
    table = pd.DataFrame(rows)
    table.to_csv(out / "model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    display = table.copy()
    display["label"] = display.run.map(
        lambda name: (
            name.replace("cnn_lstm_", "CNN-LSTM / ")
            .replace("lstm_", "LSTM / ")
            .replace("patient_history_contrastive", "History + contrastive")
            .replace("patient_history", "History")
            .replace("ecg_only", "ECG only")
            .replace("demographics", "Demographics")
            .replace("repeat_window", "Repeated window")
            .replace("persistence", "Persistence")
        )
    )
    display.set_index("label")[["mae", "rmse"]].rename(
        columns={"mae": "MAE", "rmse": "RMSE"}
    ).plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set(
        xlabel="Error (mV)",
        ylabel="",
        title="Small-data pilot: held-out patients"
        if cfg.get("subset")
        else "Held-out patient test fold",
    )
    save_figure(fig, out / "model_comparison.png")
    names = set(table.run)
    pairs = []
    for architecture in ("lstm", "cnn_lstm"):
        pairs += [
            (f"{architecture}_ecg_only", f"{architecture}_demographics", "demographics"),
            (f"{architecture}_demographics", f"{architecture}_patient_history", "history"),
            (
                f"{architecture}_patient_history",
                f"{architecture}_patient_history_contrastive",
                "contrastive",
            ),
        ]
    pairs += [
        (f"lstm_{v}", f"cnn_lstm_{v}", "architecture")
        for v in ("ecg_only", "demographics", "patient_history", "patient_history_contrastive")
    ]
    comparisons = []
    for a, b, effect in pairs:
        if a in names and b in names:
            result = bootstrap_pair(
                pd.read_csv(root / a / "per_record_metrics.csv"),
                pd.read_csv(root / b / "per_record_metrics.csv"),
                cfg["evaluation"]["bootstrap_samples"],
                cfg["training"]["seed"],
            )
            comparisons.append({"effect": effect, "A": a, "B": b, **result})
    pd.DataFrame(comparisons).to_csv(out / "paired_patient_bootstrap.csv", index=False)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(compare_runs(config).to_string(index=False))
