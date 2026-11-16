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
