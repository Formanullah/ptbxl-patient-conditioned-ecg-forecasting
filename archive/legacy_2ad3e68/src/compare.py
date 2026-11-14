"""Compare saved measurements from the current cohort without running inference."""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from config import digest, file_hash, load_config, protocol, read_json
from evaluate import save_figure
from metrics import bootstrap_pair

RESULT_COLUMNS = [
    "run", "model", "variant", "patient_macro_mae", "mae", "rmse", "mse",
    "qrs_f1", "parameters", "actual_epochs", "best_epoch",
]
PAIR_COLUMNS = [
    "A", "B", "split", "delta_patient_macro_mae_mV", "ci95_low", "ci95_high",
    "patients", "bootstrap_samples",
]


def current_manifest(cfg):
    """Read provenance metadata; report missing preprocessing without inventing counts."""
    path = Path(cfg["data"]["cache"]) / "manifest.json"
    if not path.exists():
        return None
    manifest = read_json(path)
    if manifest["protocol"] != protocol(cfg):
        raise ValueError("Configured cache belongs to another protocol; prepare a new cache")
    unsigned = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if digest(unsigned) != manifest["fingerprint"]:
        raise ValueError("Invalid cache manifest fingerprint")
    return manifest


def cohort_key(manifest):
    """Allow preprocessing ablations, while requiring identical actual patients/windows."""
    return {
        **{key: manifest[key] for key in ("metadata_sha256", "scp_sha256")},
        **{key: manifest["protocol"][key] for key in ("data", "subset")},
        "identifiers": {
            split: {key: values.get(key) for key in ("ids_sha256", "window_ids_sha256")}
            for split, values in manifest["splits"].items()
        },
    }


def comparison_controls(config):
    """Exclude the factor under comparison and operational paths from matched controls."""
    training = {
        key: value for key, value in config["training"].items()
        if key not in ("device", "threads", "num_workers")
    }
    model = {key: value for key, value in config["model"].items() if key != "fusion"}
    return digest({
        "training": training,
        "model": model,
        "preprocessing": config["preprocessing"],
        "contrastive": config["contrastive"],
    })


def collect_runs(cfg, split="val"):
    """Return matching metrics and explicit incomplete/excluded states."""
    manifest = current_manifest(cfg)
    rows, states = [], []
    root = Path(cfg["paths"]["experiments"])
    for run in sorted(root.iterdir()) if root.exists() else []:
        if not run.is_dir() or not (run / "config.json").exists():
            continue
        recorded = read_json(run / "config.json")
        if recorded.get("smoke", False):
            states.append({"run": run.name, "state": "excluded: smoke test"})
            continue
        saved = recorded["config"]
        if not (run / f"{split}_metrics.json").exists():
            state = "evaluation pending" if (run / "training_metrics.json").exists() else "training incomplete"
            states.append({"run": run.name, "state": state})
            continue
        metrics = read_json(run / f"{split}_metrics.json")
        source_manifest = Path(saved["data"]["cache"]) / "manifest.json"
        if manifest is None or not source_manifest.exists():
            states.append({"run": run.name, "state": "excluded: prepared cohort unavailable"})
            continue
        run_manifest = read_json(source_manifest)
        same_cohort = cohort_key(run_manifest) == cohort_key(manifest)
        unsigned = {key: value for key, value in run_manifest.items() if key != "fingerprint"}
        valid_provenance = (
            digest(unsigned) == run_manifest["fingerprint"] == metrics["fingerprint"]
        )
        if not same_cohort or not valid_provenance:
            states.append({"run": run.name, "state": "excluded: cohort/window/provenance mismatch"})
            continue
        checkpoint_hash = metrics.get("checkpoint_sha256")
        checkpoint = run / "best.pt"
        if checkpoint_hash and (
            not checkpoint.exists() or file_hash(checkpoint) != checkpoint_hash
        ):
            states.append({"run": run.name, "state": "evaluation stale: re-evaluate selected checkpoint"})
            continue
        states.append({"run": run.name, "state": f"{split} evaluated"})
        history_path = run / "history.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        model, training = saved["model"], saved["training"]
        baseline = "architecture" not in recorded
        row = {
            **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))},
            "run": run.name,
            "model": recorded.get("architecture", "baseline"),
            "variant": recorded.get("variant", "ecg_only"),
            "split": split,
            **{f"training_{key}": manifest["splits"]["train"][key] for key in ("patients", "records", "windows")},
            "history_seconds": saved["data"]["history_seconds"],
            "horizon_seconds": saved["data"]["horizon_seconds"],
            "filter": saved["preprocessing"]["filter"],
            "scaling": saved["preprocessing"]["scaling"],
            "fusion": model.get("fusion", "late"),
            "seed": training["seed"],
            "lr": None if baseline else training["lr"],
            "hidden": None if baseline else model["hidden"],
            "depth": None if baseline else model["depth"],
            "dropout": None if baseline else model["dropout"],
            "batch_size": None if baseline else training["batch_size"],
            "epoch_budget": 0 if baseline else training["epochs"],
            "actual_epochs": len(history),
            "runtime_seconds": float(history.seconds.sum()) if len(history) else 0.0,
            "parameters": recorded.get("parameters", 0),
            "control_key": comparison_controls(saved),
            "final_val_patient_macro_mae": float(history.val_patient_macro_mae.iloc[-1])
            if "val_patient_macro_mae" in history else None,
        }
        row["evaluated_records"] = row.pop("n_test_records", row.get("n_records"))
        row["evaluated_patients"] = row.pop("n_test_patients", row.get("n_patients"))
        rows.append(row)
    table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=RESULT_COLUMNS)
    if len(table):
        table = table.sort_values(["patient_macro_mae", "rmse", "parameters"])
    return table, pd.DataFrame(states, columns=["run", "state"])


def comparison_pairs(table):
    """Resolve planned contrast types from run metadata, independent of dataset size."""
    feature_pairs = {
        frozenset(pair) for pair in (
            ("ecg_only", "demographics"), ("demographics", "patient_history"),
            ("patient_history", "patient_history_contrastive"),
            ("expanded", "mrmr"), ("expanded", "permutation"),
        )
    }
    pairs = []
    for first, second in combinations(table.to_dict("records"), 2):
        if "baseline" in (first["model"], second["model"]):
            match = first["model"] != second["model"]
        elif first["control_key"] != second["control_key"]:
            match = False
        else:
            same_model = first["model"] == second["model"]
            same_variant = first["variant"] == second["variant"]
            same_fusion = first["fusion"] == second["fusion"]
            architecture = same_variant and same_fusion and not same_model
            features = same_model and same_fusion and frozenset((first["variant"], second["variant"])) in feature_pairs
            fusion = same_model and same_variant and not same_fusion
            match = architecture or features or fusion
        if match:
            # Stable direction: B minus A must not flip when validation ranking changes.
            pairs.append(tuple(sorted((first["run"], second["run"]))))
    return sorted(pairs)


def compare_runs(cfg, split="val"):
    import matplotlib.pyplot as plt

    if split not in ("val", "test"):
        raise ValueError("Comparison split must be val or test")
    table, states = collect_runs(cfg, split)
    out, root = [Path(cfg["paths"][key]) for key in ("outputs", "experiments")]
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{split}_comparison.csv", index=False)
    states.to_csv(out / "run_status.csv", index=False)
    pairs = []
    for first, second in comparison_pairs(table):
        paths = [root / name / f"{split}_per_record_metrics.csv" for name in (first, second)]
        if all(path.exists() for path in paths):
            result = bootstrap_pair(
                *(pd.read_csv(path) for path in paths),
                samples=cfg["evaluation"]["bootstrap_samples"],
                seed=cfg["training"]["seed"],
            )
            pairs.append({"A": first, "B": second, "split": split, **result})
    pd.DataFrame(pairs, columns=PAIR_COLUMNS).to_csv(out / f"{split}_paired_bootstrap.csv", index=False)
    if len(table):
        fig, ax = plt.subplots(figsize=(11, max(4, len(table) * 0.35)))
        table.set_index("run")[["patient_macro_mae", "rmse"]].plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set(xlabel="Error (mV)", title=f"{split}: current cohort, identical forecast windows")
        save_figure(fig, out / f"{split}_comparison.png")
    return table, states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    table, _ = compare_runs(load_config(args.config, args.experiment), args.split)
    print(table.to_string(index=False) if len(table) else "No evaluated runs. Train, then run evaluate.py.")
