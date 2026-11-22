<<<<<<< HEAD
"""Training-only grouped mRMR and validation grouped permutation importance."""

import argparse
=======
"""An isolated expanded-feature extension: train-only mRMR and validation permutation."""

>>>>>>> origin/work/manual-src-workflow
from pathlib import Path

import numpy as np
import pandas as pd
import torch

<<<<<<< HEAD
from config import file_hash, load_config, read_json, run_directory, runtime, write_json
from evaluate import predict, save_figure
from metrics import metric_summary, physical_arrays
from models import build_model
from preprocessing import ECGDataset, make_loaders, validate_cache


def feature_groups(names):
    groups = {
        "age": ["age_scaled", "age_missing"],
        "sex": ["sex_encoded", "sex_missing"],
        "history": [n for n in names if n.startswith("prior_") or n == "history_missing"],
    }
    for feature in ("height", "weight", "bmi", "heart_rate", "rr_mean", "rr_std"):
        if feature in names:
            groups[feature] = [feature, feature + "_missing"]
    return {key: value for key, value in groups.items() if value}


def discretize(values, bins=8):
    """Empirical quantile bins fitted only to the supplied training values."""
    values = np.asarray(values)
=======
from config import device_for, digest, file_hash, read_json, study_dir, write_json
from data import ECGDataset, loader
from evaluate import predict
from metrics import metric_summary
from models import build_model
from preprocessing import feature_arrays, inverse_scale


def discretize(values, bins=8):
>>>>>>> origin/work/manual-src-workflow
    unique = np.unique(values)
    if len(unique) <= bins:
        return np.searchsorted(unique, values)
    boundaries = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(boundaries, values, side="right")


<<<<<<< HEAD
def mutual_information(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((ai.max() + 1, bi.max() + 1), dtype=float)
    np.add.at(joint, (ai, bi), 1)
    joint /= joint.sum()
    independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / independent[mask])))


def mrmr_ranking(features, target, names):
    groups = feature_groups(names)
    x = np.column_stack([discretize(features[:, i]) for i in range(features.shape[1])])
    y = np.column_stack([discretize(target[:, i]) for i in range(target.shape[1])])
    relevance = {
        name: np.mean(
            [
                mutual_information(x[:, names.index(column)], y[:, j])
                for column in columns
                for j in range(y.shape[1])
            ]
        )
        for name, columns in groups.items()
    }
    selected, rows = [], []
    while len(selected) < len(groups):
        candidates = []
        for name, columns in groups.items():
            if name in selected:
                continue
            previous = sum((groups[g] for g in selected), [])
=======
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
>>>>>>> origin/work/manual-src-workflow
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
<<<<<<< HEAD
            candidates.append((float(relevance[name] - redundancy), name, redundancy))
        score, name, redundancy = max(candidates, key=lambda c: (c[0], -list(groups).index(c[1])))
        selected.append(name)
        rows.append(
            {
                "group": name,
                "rank": len(selected),
                "relevance_MI": float(relevance[name]),
                "redundancy_MI": redundancy,
                "mrmr_score": score,
            }
        )
    return pd.DataFrame(rows), groups


def mrmr(cfg, count):
    manifest = validate_cache(cfg)
    ds = ECGDataset(cfg, "train", "expanded")
    # Compute only small summary batches, not a full-population voltage copy.
    batches = []
    for start in range(0, len(ds), 128):
        target = physical_arrays(np.asarray(ds.y[start : start + 128]), manifest)
        batches.append(
            np.column_stack(
                [
                    np.sqrt(np.mean(target**2, axis=2)),
                    np.sqrt(np.mean(np.diff(target, axis=2) ** 2, axis=2)),
                ]
            )
        )
    summaries = np.concatenate(batches)
    ranking, groups = mrmr_ranking(np.asarray(ds.static), summaries, ds.feature_names)
    selected = ranking.group.iloc[:count].tolist()
    return save_selection(
        cfg,
        "mrmr",
        ranking,
        groups,
        selected,
        manifest,
        {
            "fitted_split": "train",
            "target": "24 future RMS/difference-RMS summaries in mV",
            "estimator": "8-bin training quantile mutual information; grouped MID criterion",
            "training_windows": len(ds),
            "training_patients": int(ds.metadata.patient_id.nunique()),
            "used_test": False,
            "used_validation": False,
        },
    )


def permutation(cfg, architecture, variant, count, repeats):
    manifest = validate_cache(cfg)
    run = run_directory(cfg, architecture, variant)
    recorded = read_json(run / "config.json")
    loader = make_loaders(cfg, variant, splits=("val",))["val"]
    names = loader.dataset.selected_features
    if names != loader.dataset.feature_names:
        raise ValueError("Permutation ranking requires a fitted expanded-feature model")
    if recorded["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Permutation source cache mismatch")
    device = runtime(cfg)
    model = build_model(recorded["config"], architecture, len(names)).to(device)
    model.load_state_dict(
        torch.load(run / "best.pt", map_location=device, weights_only=True)["model"]
    )
    pred, true = predict(model, loader, device)
    true = physical_arrays(true, manifest, cfg["data"].get("layout"))
    patients = loader.dataset.patients
    baseline = metric_summary(
        physical_arrays(pred, manifest, cfg["data"].get("layout")), true, patients
    )["patient_macro_mae"]
    original = np.asarray(loader.dataset.static).copy()
    groups = feature_groups(names)
    rng = np.random.default_rng(cfg["training"]["seed"])
    unique = np.unique(patients)
    first = {p: np.flatnonzero(patients == p)[0] for p in unique}
    rows = []
    for name, columns in groups.items():
        indices = [names.index(c) for c in columns]
        changes = []
        for _ in range(repeats):
            donors = dict(zip(unique, rng.permutation(unique), strict=True))
            source = np.array([first[donors[p]] for p in patients])
            shuffled = original.copy()
            shuffled[:, indices] = original[source][:, indices]
            loader.dataset.static = shuffled
            pred, _ = predict(model, loader, device)
            score = metric_summary(
                physical_arrays(pred, manifest, cfg["data"].get("layout")), true, patients
            )["patient_macro_mae"]
            changes.append(score - baseline)
        rows.append(
            {
                "group": name,
                "importance_mV": float(np.mean(changes)),
                "repeat_std_mV": float(np.std(changes)),
                "repeats": repeats,
            }
        )
    loader.dataset.static = original
    ranking = (
        pd.DataFrame(rows).sort_values("importance_mV", ascending=False).reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    selected = ranking.loc[ranking.importance_mV > 0, "group"].iloc[:count].tolist()
    return save_selection(
        cfg,
        "permutation",
        ranking,
        groups,
        selected,
        manifest,
        {
            "fitted_split": "validation ranking of training-fitted model",
            "source_checkpoint": str(run / "best.pt"),
            "checkpoint_sha256": file_hash(run / "best.pt"),
            "baseline_patient_macro_mae_mV": baseline,
            "used_test": False,
            "permutation_unit": "patient donors; joint group and flags; first donor record",
            "interpretation": "conditional predictive association, not causal; correlated features can substitute",
        },
    )


def save_selection(cfg, method, ranking, groups, selected, manifest, details):
    import matplotlib.pyplot as plt

    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    names = manifest["static_statistics"]["features"]
    chosen = set(sum((groups[g] for g in selected), []))
    artifact = {
        "method": method,
        "fingerprint": manifest["fingerprint"],
        "selected_groups": selected,
        "selected_features": [n for n in names if n in chosen],
        "ranking": ranking.to_dict("records"),
        "groups": groups,
        **details,
    }
    write_json(out / f"{method}.json", artifact)
    ranking.to_csv(out / f"{method}_ranking.csv", index=False)
    value = "importance_mV" if method == "permutation" else "mrmr_score"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ranking.group, ranking[value])
    ax.invert_yaxis()
    ax.set(xlabel=value, title=f"{method}: grouped static/observed-feature ranking")
    save_figure(fig, out / f"{method}_ranking.png")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--method", choices=["mrmr", "permutation"], required=True)
    parser.add_argument("--model", choices=["lstm", "cnn_lstm"], default="lstm")
    parser.add_argument("--variant", default="expanded")
    parser.add_argument("--count", type=int, default=3, help="Maximum selected feature groups")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.count < 1 or args.repeats < 1:
        parser.error("count and repeats must be positive")
    cfg = load_config(args.config, args.experiment)
    result = (
        mrmr(cfg, args.count)
        if args.method == "mrmr"
        else permutation(cfg, args.model, args.variant, args.count, args.repeats)
    )
    print(result)
=======
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
>>>>>>> origin/work/manual-src-workflow
