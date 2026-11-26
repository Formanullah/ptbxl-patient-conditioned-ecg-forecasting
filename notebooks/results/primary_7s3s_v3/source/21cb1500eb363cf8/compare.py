<<<<<<< HEAD
﻿"""Compare saved measurements from the current cohort without running inference."""

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
=======
"""Aggregate only traceable measurements and identify controlled comparisons."""

from itertools import combinations

import pandas as pd

from config import digest, file_hash, protocol, read_json, study_dir
from evaluate import verify_evaluation
from metrics import bootstrap_pair

RESULT_COLUMNS = [
    "name",
    "state",
    "split",
    "architecture",
    "preprocessing",
    "feature_set",
    "mae_mV",
    "rmse_mV",
    "patient_macro_mae_mV",
    "mse_mV2",
    "r2",
    "pearson_mean",
    "records",
    "patients",
    "seed",
    "lr",
    "hidden",
    "epochs_completed",
    "best_epoch",
    "runtime_seconds",
]


def collect(cfg):
    root = study_dir(cfg) / cfg["run_mode"]
    rows, lead_frames, records = [], [], {}
    for directory in sorted((root / "checkpoints").glob("*")):
        if not (directory / "run.json").exists():
            continue
        record = read_json(directory / "run.json")
        row = {
            "name": directory.name,
            "state": "training incomplete",
            "architecture": record["architecture"],
            "preprocessing": record["preprocessing"],
            "feature_set": record["feature_set"],
            "parameters": record["parameters"],
            "seed": record["config"]["training"]["seed"],
            "lr": record["config"]["training"]["lr"],
            "hidden": record["config"]["model"]["hidden"],
            "smoke": record["smoke"],
        }
        if (directory / "status.json").exists():
            state = read_json(directory / "status.json")
            row.update(
                state=state["state"],
                epochs_completed=state["epochs_completed"],
                runtime_seconds=state["runtime_seconds"],
            )
        for split in ("val", "test"):
            path = root / "metrics" / directory.name / f"{split}.json"
            if not path.exists():
                continue
            metrics = read_json(path)
            try:
                verify_evaluation(path.parent, split)
                evidence_valid = True
            except (ValueError, FileNotFoundError):
                evidence_valid = False
            valid = (
                record["signature"] == digest(record["scientific"])
                and evidence_valid
                and metrics["cohort"] == record["scientific"]["cohort"]
                and metrics["run_signature"] == record["signature"]
                and metrics["protocol"] == protocol(cfg)
                and (directory / "best.pt").exists()
                and metrics["checkpoint_sha256"] == file_hash(directory / "best.pt")
                and (directory / "cohort_manifest.json").exists()
                and (directory / "scaler.json").exists()
                and file_hash(directory / "scaler.json") == record["scientific"]["scaler_sha256"]
            )
            entry = {**row, "split": split}
            if not valid:
                entry["state"] = "incompatible or unavailable provenance"
            elif row["state"] != "completed":
                entry["state"] = "training incomplete"
            elif metrics["smoke"]:
                if not metrics["synthetic"]:
                    entry.update(metrics)
                entry["state"] = "software smoke check passed; excluded from research comparison"
            else:
                entry.update(metrics, state="measured")
                lead_frames.append(
                    pd.read_csv(path.with_name(f"{split}_leads.csv")).assign(
                        name=directory.name, split=split
                    )
                )
                records[(directory.name, split)] = record
            rows.append(entry)
        if not any(
            (root / "metrics" / directory.name / f"{s}.json").exists() for s in ("val", "test")
        ):
            rows.append(row)
    for name in ("persistence", "repeat_window"):
        for split in ("val", "test"):
            path = root / "metrics" / name / f"{split}.json"
            if path.exists():
                result = read_json(path)
                try:
                    verify_evaluation(path.parent, split)
                except (ValueError, FileNotFoundError):
                    rows.append(
                        {
                            "name": name,
                            "split": split,
                            "state": "incompatible or unavailable provenance",
                        }
                    )
                    continue
                if result["smoke"] and not result["synthetic"]:
                    rows.append(
                        {
                            **result,
                            "state": "software smoke check passed; excluded from research comparison",
                        }
                    )
                elif not result["smoke"] and result["protocol"] == protocol(cfg):
                    rows.append(
                        {**result, "state": "measured", "epochs_completed": 0, "best_epoch": None}
                    )
                    lead_frames.append(
                        pd.read_csv(path.with_name(f"{split}_leads.csv")).assign(
                            name=name, split=split
                        )
                    )
    table = pd.DataFrame(rows).reindex(
        columns=list(dict.fromkeys(RESULT_COLUMNS + [k for r in rows for k in r]))
    )
    leads = (
        pd.concat(lead_frames, ignore_index=True)
        if lead_frames
        else pd.DataFrame(columns=["name", "split", "lead", "mae_mV", "rmse_mV", "pearson"])
    )
    return table, leads, records


def paired_comparisons(cfg, table, records):
    output = []
    candidates = table.loc[table.state.eq("measured")].to_dict("records")
    for a, b in combinations(candidates, 2):
        if (
            a["split"] != b["split"]
            or a["cohort"] != b["cohort"]
            or a["target_sha256"] != b["target_sha256"]
        ):
            continue
        ra, rb = records.get((a["name"], a["split"])), records.get((b["name"], b["split"]))
        contrast = None
        if ra is None or rb is None:
            if (ra is None) != (rb is None):
                contrast = "neural pipeline versus raw-history baseline"
                if rb is None:
                    a, b = b, a
        else:
            sa, sb = ra["scientific"], rb["scientific"]
            controls = (
                "cohort",
                "model",
                "training",
                "budget",
                "source",
                "versions",
                "scaler_sha256",
            )
            same_model = a["architecture"] == b["architecture"]
            same_features = sa["features"] == sb["features"]
            same_filter = a["preprocessing"] == b["preprocessing"]
            matched = all(sa[k] == sb[k] for k in controls)
            if not matched:
                if cfg["run_mode"] == "full" and not same_model and same_filter:
                    contrast = "selected pipelines (architecture and recipe may differ)"
                elif (
                    same_model
                    and same_filter
                    and same_features
                    and sa["model"] == sb["model"]
                    and (
                        {k: v for k, v in sa["training"].items() if k != "lr"}
                        == {k: v for k, v in sb["training"].items() if k != "lr"}
                    )
                    and all(sa[k] == sb[k] for k in controls if k != "training")
                ):
                    contrast = "learning-rate comparison"
            elif same_model and same_features and not same_filter:
                contrast = "P0 versus P1"
                if a["preprocessing"] == "P1":
                    a, b = b, a
            elif same_model and same_filter and not same_features:
                contrast = "static-feature ablation"
                priority = {"F0": 0, "F1": 1, "expanded": 2, "mrmr": 3, "permutation": 3}
                if priority.get(a["feature_set"], 2) > priority.get(b["feature_set"], 2):
                    a, b = b, a
            elif not same_model and same_features and same_filter:
                contrast = "matched architecture comparison"
        if contrast is None:
            continue
        root = study_dir(cfg) / cfg["run_mode"] / "metrics"
        first = pd.read_csv(root / a["name"] / f"{a['split']}_records.csv")
        second = pd.read_csv(root / b["name"] / f"{b['split']}_records.csv")
        comparison = bootstrap_pair(
            first, second, cfg["evaluation"]["bootstrap_samples"], cfg["training"]["seed"]
        )
        denominator = a["patient_macro_mae_mV"]
        output.append(
            {
                "A": a["name"],
                "B": b["name"],
                "split": a["split"],
                "contrast": contrast,
                "percentage_B_minus_A": 100
                * (b["patient_macro_mae_mV"] - denominator)
                / denominator
                if denominator > 1e-12
                else None,
                **comparison,
            }
        )
    return pd.DataFrame(
        output,
        columns=[
            "A",
            "B",
            "split",
            "contrast",
            "percentage_B_minus_A",
            "difference_B_minus_A_mV",
            "ci95_low",
            "ci95_high",
            "patients",
            "bootstrap_samples",
            "conclusion",
        ],
    )
>>>>>>> origin/work/manual-src-workflow
