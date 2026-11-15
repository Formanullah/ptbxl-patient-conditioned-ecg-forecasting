"""Training-only grouped mRMR and validation grouped permutation importance."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    unique = np.unique(values)
    if len(unique) <= bins:
        return np.searchsorted(unique, values)
    boundaries = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(boundaries, values, side="right")


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
