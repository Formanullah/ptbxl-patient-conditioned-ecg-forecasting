from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

from config import CLASSES, digest, file_hash, read_json, write_json


def filter_history(history, cfg):
    """Accept [time, lead]; no future samples enter this function."""
    p = cfg["preprocessing"]
    sos = signal.butter(
        p["order"], [p["low_hz"], p["high_hz"]], fs=100, btype="bandpass", output="sos"
    )
    zi = signal.sosfilt_zi(sos)[:, :, None] * history[0][None, None, :]
    result, _ = signal.sosfilt(sos, history, axis=0, zi=zi)
    return result.astype(np.float32)


def waveform_scaler(waves, history, batch_size=128):
    total = np.zeros(12, dtype=np.float64)
    squares = total.copy()
    count = 0
    for start in range(0, len(waves), batch_size):
        x = np.asarray(waves[start : start + batch_size, :history], dtype=np.float64)
        total += x.sum(axis=(0, 1))
        squares += np.square(x).sum(axis=(0, 1))
        count += x.shape[0] * x.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    return {
        "mean": mean.tolist(),
        "std": np.where(std > 1e-8, std, 1).tolist(),
        "fitted_split": "train",
        "fitted_samples_per_lead": count,
        "definition": "raw historical inputs only; population standard deviation",
    }


def inverse_scale(x, scaler):
    return np.asarray(x) * np.asarray(scaler["std"]) + np.asarray(scaler["mean"])


def prepare_transforms(cache, cfg, preprocessing, *, include_test=False):
    cache = Path(cache)
    base = read_json(cache / "manifest.json")
    history = cfg["data"]["history"]
    scaler_path = cache / "scaler.json"
    if not scaler_path.exists():
        waves = np.load(cache / "waves_train.npy", mmap_mode="r")
        write_json(scaler_path, {**waveform_scaler(waves, history), "cohort": base["cohort"]})
        write_json(cache / "scaler_receipt.json", {"sha256": file_hash(scaler_path)})
    if read_json(cache / "scaler_receipt.json")["sha256"] != file_hash(scaler_path):
        raise ValueError("Saved training scaler changed")
    scaler = read_json(scaler_path)
    if scaler.get("cohort") != base["cohort"]:
        raise ValueError("Scaler belongs to a different cohort")
    if preprocessing not in ("P0", "P1"):
        raise ValueError("preprocessing must be P0 or P1")
    if preprocessing == "P0":
        return scaler
    if include_test:
        from evaluate import verify_lock

        verify_lock(cfg, cache)
    identity = digest({"cohort": base["cohort"], "settings": cfg["preprocessing"]})
    splits = ("train", "val", "test") if include_test else ("train", "val")
    for split in splits:
        raw_path = cache / f"waves_{split}.npy"
        if not raw_path.exists():
            continue
        path = cache / f"P1_{identity[:12]}_{split}.npy"
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists():
            info = read_json(receipt)
            if info["identity"] != identity or info["sha256"] != file_hash(path):
                raise ValueError("Filtered cache changed; use a fresh cache directory")
            continue
        waves = np.load(raw_path, mmap_mode="r")
        output = np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=(len(waves), history, 12)
        )
        for i in range(len(waves)):
            output[i] = filter_history(waves[i, :history], cfg)
        output.flush()
        del output
        write_json(
            receipt,
            {
                "identity": identity,
                "sha256": file_hash(path),
                "target": "unchanged raw future",
                "scaler_sha256": file_hash(scaler_path),
            },
        )
    return scaler


def filtered_path(cache, cfg, split):
    base = read_json(Path(cache) / "manifest.json")
    key = digest({"cohort": base["cohort"], "settings": cfg["preprocessing"]})[:12]
    return Path(cache) / f"P1_{key}_{split}.npy"


def prior_diagnoses(frame, root):
    """Group equal timestamps before updating history, excluding self and same-time labels."""
    mapping = pd.read_csv(Path(root) / "scp_statements.csv", index_col=0)
    mapping = mapping.loc[mapping.diagnostic.eq(1), "diagnostic_class"].dropna().to_dict()
    labels = np.zeros((len(frame), 5), dtype=np.float32)
    for i, codes in enumerate(frame.scp_codes):
        parsed = ast.literal_eval(codes)
        if not isinstance(parsed, dict):
            raise ValueError("SCP annotations must be dictionaries")
        classes = {mapping.get(k) for k in parsed}
        labels[i] = [name in classes for name in CLASSES]
    prior = np.zeros((len(frame), 6), dtype=np.float32)
    prior[:, -1] = 1
    work = frame[["patient_id", "recording_date"]].copy()
    work["position"] = np.arange(len(frame))
    work["date"] = pd.to_datetime(work.recording_date, errors="coerce")
    edges = []
    for _, patient in work.groupby("patient_id", sort=False):
        known, previous = np.zeros(5, dtype=np.float32), []
        for date, same_time in patient.dropna(subset=["date"]).groupby("date", sort=True):
            indices = same_time.position.to_numpy(dtype=int)
            prior[indices, :5] = known
            prior[indices, -1] = float(not previous)
            for i in indices:
                for j, previous_date in previous:
                    if not previous_date < date:
                        raise ValueError("Non-past diagnosis entered historical features")
                    edges.append((int(frame.iloc[j].ecg_id), int(frame.iloc[i].ecg_id)))
            known = np.maximum(known, labels[indices].max(axis=0))
            previous.extend((i, date) for i in indices)
    return prior, {
        "passed": True,
        "strict_prior_edges": len(edges),
        "edge_hash": digest(edges),
        "available_records": int((prior[:, -1] == 0).sum()),
        "missing_dates": int(work.date.isna().sum()),
        "interpretation": "prior ECG annotation proxies, not a clinical problem list",
    }


def observed_rr(history):
    from wfdb import processing

    if np.ptp(history[:, 1]) < 1e-8:
        return [np.nan] * 3
    try:
        peaks = processing.xqrs_detect(history[:, 1], fs=100, verbose=False)
        rr = np.diff(peaks) / 100
        return [60 / rr.mean(), rr.mean(), rr.std()] if len(rr) >= 2 else [np.nan] * 3
    except (ValueError, IndexError, FloatingPointError):
        return [np.nan] * 3


def feature_arrays(cfg, cache, feature_set, splits=None):
    """Return train-fitted static arrays; expensive observed features are extension-only."""
    cache = Path(cache)
    splits = list(splits) if splits is not None else ["train", "val"]
    if "test" in splits:
        from evaluate import verify_lock

        verify_lock(cfg, cache)
    frames = {s: pd.read_csv(cache / f"metadata_{s}.csv") for s in splits}
    full = pd.concat([frames[s].assign(split=s) for s in splits], ignore_index=True)
    train = full.split.eq("train").to_numpy()
    base = read_json(cache / "manifest.json")
    if feature_set == "F0":
        return (
            {s: np.zeros((len(frames[s]), 0), dtype=np.float32) for s in splits},
            [],
            {"groups": {}, "statistics": {}},
        )
    raw, names, groups, stats = [], [], {}, {}

    def continuous(name, values):
        values = np.asarray(values, dtype=float)
        observed = values[train & np.isfinite(values)]
        center = float(np.median(observed)) if len(observed) else 0.0
        std = float(observed.std()) if len(observed) and observed.std() > 1e-8 else 1.0
        missing = ~np.isfinite(values)
        raw.extend([(np.where(missing, center, values) - center) / std, missing])
        names.extend([name, name + "_missing"])
        groups[name] = [name, name + "_missing"]
        stats[name] = {"median": center, "std": std, "training_observations": len(observed)}

    age = pd.to_numeric(full.age, errors="coerce").where(lambda a: a.between(0, 120))
    continuous("age", age)
    sex = pd.to_numeric(full.sex, errors="coerce")
    valid = sex.isin([0, 1])
    observed_sex = sex[train & valid]
    fill = float(observed_sex.mode().iloc[0]) if len(observed_sex) else 0.0
    raw.extend([sex.where(valid, fill).to_numpy(), (~valid).to_numpy()])
    names.extend(["sex", "sex_missing"])
    groups["sex"] = ["sex", "sex_missing"]
    stats["sex"] = {"training_mode": fill}
    prior_audit = None
    if feature_set == "F2" or (feature_set == "expanded" and cfg["features"]["prior_history"]):
        if not cfg["features"]["prior_history"]:
            raise ValueError("Enable and audit prior_history explicitly before F2")
        prior, prior_audit = prior_diagnoses(full, cfg["data"]["root"])
        prior_names = ["prior_" + c for c in CLASSES] + ["history_missing"]
        raw.extend(prior.T)
        names.extend(prior_names)
        groups["history"] = prior_names
    if feature_set == "expanded":
        height = pd.to_numeric(
            full.get("height", pd.Series(np.nan, index=full.index)), errors="coerce"
        )
        weight = pd.to_numeric(
            full.get("weight", pd.Series(np.nan, index=full.index)), errors="coerce"
        )
        height, weight = (
            height.where(height.between(50, 250)),
            weight.where(weight.between(10, 400)),
        )
        for name, values in (
            ("height", height),
            ("weight", weight),
            ("bmi", weight / (height / 100) ** 2),
        ):
            continuous(name, values)
        rr_key = digest(
            {"cohort": base["cohort"], "splits": splits, "history": cfg["data"]["history"]}
        )
        rr_path = cache / f"observed_rr_{rr_key[:12]}.npy"
        receipt = rr_path.with_suffix(".json")
        if receipt.exists() and read_json(receipt).get("identity") == rr_key:
            if file_hash(rr_path) != read_json(receipt)["sha256"]:
                raise ValueError("Observed RR cache changed")
            rr = np.load(rr_path)
        else:
            values = []
            for split in splits:
                waves = np.load(cache / f"waves_{split}.npy", mmap_mode="r")
                values.extend(observed_rr(w[: cfg["data"]["history"]]) for w in waves)
            rr = np.asarray(values)
            np.save(rr_path, rr)
            write_json(receipt, {"identity": rr_key, "sha256": file_hash(rr_path)})
        for i, name in enumerate(("heart_rate", "rr_mean", "rr_std")):
            continuous(name, rr[:, i])
    matrix = np.column_stack(raw).astype(np.float32)
    arrays = {s: matrix[full.split.eq(s)].copy() for s in splits}
    return (
        arrays,
        names,
        {
            "groups": groups,
            "statistics": stats,
            "prior_audit": prior_audit,
            "fitted_split": "train",
            "cohort": base["cohort"],
        },
    )
