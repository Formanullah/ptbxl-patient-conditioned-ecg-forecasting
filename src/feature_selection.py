"""An isolated expanded-feature extension: train-only mRMR and validation permutation."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import device_for, digest, file_hash, read_json, study_dir, write_json
from data import ECGDataset, loader
from evaluate import predict
from metrics import metric_summary
from models import build_model
from preprocessing import feature_arrays, inverse_scale


def discretize(values, bins=8):
    unique = np.unique(values)
    if len(unique) <= bins:
        return np.searchsorted(unique, values)
    boundaries = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(boundaries, values, side="right")


def mutual_information(first, second):
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    independent = joint.sum(0)[None, :] * joint.sum(1)[:, None]
    use = joint > 0
    return float(np.sum(joint[use] * np.log(joint[use] / independent[use])))


def mrmr_rank(features, target, names, groups):
    x = np.column_stack([discretize(features[:, i]) for i in range(features.shape[1])])
    y = np.column_stack([discretize(target[:, i]) for i in range(target.shape[1])])
    relevance = {
        g: float(
            np.mean(
                [
                    mutual_information(x[:, names.index(c)], y[:, j])
                    for c in columns
                    for j in range(y.shape[1])
                ]
            )
        )
        for g, columns in groups.items()
    }
    selected, rows = [], []
    while len(selected) < len(groups):
        scores = []
        previous = [c for g in selected for c in groups[g]]
        for group, columns in groups.items():
            if group in selected:
                continue
            redundancy = (
                float(
                    np.mean(
                        [
                            mutual_information(x[:, names.index(a)], x[:, names.index(b)])
                            for a in columns
                            for b in previous
                        ]
                    )
                )
                if previous
                else 0.0
            )
            scores.append((relevance[group] - redundancy, group, redundancy))
        score, group, redundancy = max(scores, key=lambda r: (r[0], -list(groups).index(r[1])))
        selected.append(group)
        rows.append(
            {
                "group": group,
                "rank": len(selected),
                "relevance_MI": relevance[group],
                "redundancy_MI": redundancy,
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def save_selection(cfg, cache, method, ranking, names, groups, selected, details):
    path = (
        study_dir(cfg)
        / cfg["run_mode"]
        / "feature_selection"
        / details.get("architecture", "shared")
    )
    path.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(path / f"{method}_ranking.csv", index=False)
    chosen = {c for group in selected for c in groups[group]}
    artifact = {
        "method": method,
        "cohort": read_json(Path(cache) / "manifest.json")["cohort"],
        "selected_groups": selected,
        "selected_features": [n for n in names if n in chosen],
        "groups": groups,
        "used_test": False,
        **details,
    }
    artifact["signature"] = digest(artifact)
    write_json(path / f"{method}.json", artifact)
    return artifact


def mrmr(cfg, cache):
    # No validation/test arrays or outcomes are opened for this ranking.
    arrays, names, info = feature_arrays(cfg, cache, "expanded", splits=("train",))
    waves = np.load(Path(cache) / "waves_train.npy", mmap_mode="r")
    h = cfg["data"]["history"]
    summaries = []
    for start in range(0, len(waves), 128):
        target = np.asarray(waves[start : start + 128, h:], dtype=np.float64)
        summaries.append(
            np.column_stack(
                [np.sqrt((target**2).mean(1)), np.sqrt((np.diff(target, axis=1) ** 2).mean(1))]
            )
        )
    ranking = mrmr_rank(arrays["train"], np.concatenate(summaries), names, info["groups"])
    selected = ranking.group.iloc[: cfg["features"]["selected_groups"]].tolist()
    return save_selection(
        cfg,
        cache,
        "mrmr",
        ranking,
        names,
        info["groups"],
        selected,
        {
            "fitted_split": "train",
            "target": "24 future RMS/difference-RMS summaries, training only",
            "estimator": "8-bin training quantile grouped MID",
            "used_validation_outcomes": False,
        },
    )


def permutation(cfg, cache, run):
    run = Path(run)
    record = read_json(run / "run.json")
    if record["feature_set"] != "expanded":
        raise ValueError("Permutation ranking needs a trained expanded-feature reference")
    arrays, names, info = feature_arrays(cfg, cache, "expanded")
    if names != record["selected_features"]:
        raise ValueError("Expanded feature order changed")
    dataset = ECGDataset(cache, cfg, "val", record["preprocessing"], arrays["val"])
    batches, device = loader(dataset, cfg), device_for(cfg)
    model = build_model(
        record["architecture"], len(names), cfg["data"]["horizon"], **record["config"]["model"]
    ).to(device)
    state = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    scaler = read_json(Path(cache) / "scaler.json")
    prediction, target = predict(model, batches, device)
    target = inverse_scale(target, scaler)
    patients = dataset.metadata.patient_id.to_numpy()
    baseline = metric_summary(inverse_scale(prediction, scaler), target, patients)[0][
        "patient_macro_mae_mV"
    ]
    rng, original = np.random.default_rng(cfg["training"]["seed"]), dataset.static.copy()
    unique = np.unique(patients)
    first = {p: np.flatnonzero(patients == p)[0] for p in unique}
    rows = []
    try:
        for group, columns in info["groups"].items():
            indices, changes = [names.index(c) for c in columns], []
            for _ in range(cfg["features"]["permutation_repeats"]):
                donors = dict(zip(unique, rng.permutation(unique), strict=True))
                source = [first[donors[p]] for p in patients]
                shuffled = original.copy()
                shuffled[:, indices] = original[source][:, indices]
                dataset.static = shuffled
                prediction, _ = predict(model, batches, device)
                score = metric_summary(inverse_scale(prediction, scaler), target, patients)[0][
                    "patient_macro_mae_mV"
                ]
                changes.append(score - baseline)
            rows.append(
                {
                    "group": group,
                    "importance_mV": float(np.mean(changes)),
                    "repeat_std_mV": float(np.std(changes)),
                }
            )
    finally:
        dataset.static = original
    ranking = pd.DataFrame(rows).sort_values(["importance_mV", "group"], ascending=[False, True])
    selected = (
        ranking.loc[ranking.importance_mV > 0, "group"]
        .iloc[: cfg["features"]["selected_groups"]]
        .tolist()
    )
    return save_selection(
        cfg,
        cache,
        "permutation",
        ranking,
        names,
        info["groups"],
        selected,
        {
            "architecture": record["architecture"],
            "fitted_split": "val",
            "source_checkpoint_sha256": file_hash(run / "best.pt"),
            "permutation_unit": "patient donors; first donor record; variables and flags moved together",
            "limitation": "first-record donor approximates within-patient variation; correlated variables can substitute",
            "interpretation": "predictive association, not causality",
        },
    )
