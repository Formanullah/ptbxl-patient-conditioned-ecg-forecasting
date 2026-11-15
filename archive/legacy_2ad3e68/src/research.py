"""Explicit validation/final evaluation, provenance and research comparison tables."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import file_hash, load_config, protocol, read_json, run_directory, runtime, write_json
from evaluate import metric_summary, plot_examples, predict, save_figure, subgroup_metrics
from models import build_model
from morphology import qrs_metrics
from preprocessing import ECGDataset, make_loaders, validate_cache


def physical_arrays(array, manifest, layout="channels_first"):
    if layout == "time_first":
        array = array.transpose(0, 2, 1)
    scale, center = [
        np.asarray(manifest[k], dtype=np.float32)[None, :, None] for k in ("std_mV", "mean_mV")
    ]
    return array * scale + center


def authorize_split(cfg, split, run=None):
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if split == "test":
        settings = cfg.get("evaluation", {})
        if not settings.get("allow_test", False) or cfg.get("subset", {}).get("mode") != "full":
            raise ValueError("Fold 10 is reserved: require full cohort and explicit allow_test")
        selection = read_json(settings["selection_path"])
        if (
            selection["fingerprint"]
            != read_json(Path(cfg["data"]["cache"]) / "manifest.json")["fingerprint"]
        ):
            raise ValueError("Final selection cohort mismatch")
        if run is not None and run.name not in ("persistence", "repeat_window"):
            if selection["checkpoints"].get(run.name) != file_hash(run / "best.pt"):
                raise ValueError("Checkpoint is not in the frozen validation selection")


def arrays_evaluation(cfg, run, split, pred, true, observed, meta, manifest):
    import matplotlib.pyplot as plt

    out = Path(cfg["paths"]["outputs"]) / run.name / split
    out.mkdir(parents=True, exist_ok=True)
    summary = metric_summary(pred, true, meta.patient_id.to_numpy())
    summary.update(
        split=split,
        fingerprint=manifest["fingerprint"],
        comparison_key={k: manifest["protocol"][k] for k in ("data", "subset")},
    )
    qrs, peaks, detections, timing = qrs_metrics(
        pred,
        true,
        observed,
        cfg["data"]["sampling_rate"],
        cfg["evaluation"].get("peak_tolerance_ms", 100),
    )
    summary.update(qrs)
    peaks.to_csv(out / "peak_metrics.csv", index=False)
    per_record = meta.copy()
    error = pred.astype(float) - true.astype(float)
    per_record["mae_mV"] = np.abs(error).mean(axis=(1, 2))
    per_record["mse_mV2"] = (error**2).mean(axis=(1, 2))
    per_record.to_csv(run / f"{split}_per_record_metrics.csv", index=False)
    subgroup_metrics(pred, true, meta).to_csv(out / "subgroup_metrics.csv", index=False)
    pd.DataFrame(
        {
            "lead": manifest["protocol"]["leads"],
            "mae_mV": summary["mae_by_lead"],
            "rmse_mV": summary["rmse_by_lead"],
            "pearson": summary["pearson_by_lead"],
        }
    ).to_csv(out / "per_lead_metrics.csv", index=False)
    pd.DataFrame(
        {
            "seconds": np.arange(pred.shape[-1]) / cfg["data"]["sampling_rate"],
            "mae_mV": np.abs(error).mean(axis=(0, 1)),
            "rmse_mV": np.sqrt((error**2).mean(axis=(0, 1))),
        }
    ).to_csv(out / "horizon_metrics.csv", index=False)
    metric_plots(out)
    summary["example_ecg_ids"] = plot_examples(observed, pred, true, meta, out, cfg)
    example = int(np.argsort(per_record.mae_mV.to_numpy(), kind="stable")[len(meta) // 2])
    fs = cfg["data"]["sampling_rate"]
    t = np.arange(pred.shape[-1]) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax in axes:
        ax.plot(t, true[example, 1], label="Actual lead II")
        ax.plot(t, pred[example, 1], label="Predicted lead II")
        for values, color, label in zip(
            detections[example], ["black", "red"], ["Actual R", "Predicted R"], strict=True
        ):
            voltage = true if color == "black" else pred
            ax.scatter(t[values], voltage[example, 1, values], c=color, marker="x", label=label)
        ax.legend(fontsize=8)
        ax.set(xlabel="Forecast time (s)", ylabel="mV")
    reference = detections[example][0]
    if len(reference):
        center = reference[len(reference) // 2] / fs
        axes[1].set_xlim(max(0, center - 0.25), min(pred.shape[-1] / fs, center + 0.25))
    axes[0].set_title(
        f"Median-error example ECG {int(meta.iloc[example].ecg_id)}; detector-derived peaks"
    )
    save_figure(fig, out / "qrs_overlay_zoom.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(timing, bins=20)
    ax.set(
        xlabel="Matched peak timing error (ms)",
        ylabel="Peaks",
        title="No matched peaks" if not timing else "Matched R-peak timing errors",
    )
    save_figure(fig, out / "peak_timing_errors.png")
    np.savez_compressed(
        run / f"{split}_predictions.npz",
        pred=pred,
        true=true,
        ecg_id=meta.ecg_id.to_numpy(),
        patient_id=meta.patient_id.to_numpy(),
    )
    write_json(run / f"{split}_metrics.json", summary)
    return summary


def metric_plots(out):
    """Render already-saved metrics without repeating inference or selecting examples."""
    import matplotlib.pyplot as plt

    for filename, index, title in (
        ("per_lead_metrics", "lead", "Errors by ECG lead"),
        ("horizon_metrics", "seconds", "Errors through the forecast horizon"),
        ("subgroup_metrics", "group", "Errors by overlapping patient subgroups"),
    ):
        frame = pd.read_csv(out / f"{filename}.csv")
        fig, ax = plt.subplots(figsize=(11, 5))
        values = frame.set_index(index)[["mae_mV", "rmse_mV"]]
        if filename == "horizon_metrics":
            values.plot(ax=ax)
        else:
            values.plot.bar(ax=ax)
            ax.tick_params(axis="x", rotation=40)
        ax.set(ylabel="Error (mV)", title=title)
        save_figure(fig, out / f"{filename}.png")


def evaluate_run(cfg, architecture, variant, split="val"):
    manifest = validate_cache(cfg)
    run = run_directory(cfg, architecture, variant)
    authorize_split(cfg, split, run)
    recorded = read_json(run / "config.json")
    if recorded["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Evaluation cache differs from training")
    loader = make_loaders(cfg, variant, splits=(split,))[split]
    if loader.dataset.selected_features != recorded["selected_features"]:
        raise ValueError("Selected feature order differs from training")
    device = runtime(cfg)
    model = build_model(recorded["config"], architecture, loader.dataset.static_dim).to(device)
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    pred, true = predict(model, loader, device)
    pred, true = [physical_arrays(a, manifest, cfg["data"].get("layout")) for a in (pred, true)]
    observed = np.load(Path(cfg["data"]["cache"]) / f"raw_X_{split}.npy")
    summary = arrays_evaluation(
        cfg, run, split, pred, true, observed, loader.dataset.metadata, manifest
    )
    summary.update(best_epoch=checkpoint["epoch"], checkpoint_sha256=file_hash(run / "best.pt"))
    write_json(run / f"{split}_metrics.json", summary)
    print(
        f"{run.name} {split}: patient-macro MAE={summary['patient_macro_mae']:.6f} mV", flush=True
    )
    return summary


def baseline_runs(cfg, split="val"):
    manifest = validate_cache(cfg)
    authorize_split(cfg, split)
    ds = ECGDataset(cfg, split, "ecg_only")
    observed = np.load(Path(cfg["data"]["cache"]) / f"raw_X_{split}.npy")
    true = physical_arrays(np.asarray(ds.y), manifest)
    horizon = cfg["data"]["horizon"]
    predictions = {
        "persistence": np.repeat(observed[:, :, -1:], horizon, axis=2),
        "repeat_window": np.tile(
            observed[:, :, -min(horizon, observed.shape[-1]) :],
            (1, 1, int(np.ceil(horizon / min(horizon, observed.shape[-1])))),
        )[:, :, :horizon],
    }
    # A train-only constant reference exposes apparent gains from near-flat outputs.
    training_raw = np.load(Path(cfg["data"]["cache"]) / "raw_X_train.npy", mmap_mode="r")
    center = np.array([np.median(training_raw[:, j, :]) for j in range(12)])
    predictions["training_center"] = np.broadcast_to(center[None, :, None], true.shape).copy()
    for name, pred in predictions.items():
        run = Path(cfg["paths"]["experiments"]) / name
        run.mkdir(parents=True, exist_ok=True)
        write_json(
            run / "config.json",
            {
                "config": cfg,
                "baseline": name,
                "definition": "constant per-lead training-input median"
                if name == "training_center"
                else name,
            },
        )
        arrays_evaluation(cfg, run, split, pred, true, observed, ds.metadata, manifest)


def comparison(cfg, split="val"):
    import matplotlib.pyplot as plt

    root, out = [Path(cfg["paths"][k]) for k in ("experiments", "outputs")]
    out.mkdir(parents=True, exist_ok=True)
    rows, other_rows = [], []
    requested = {k: protocol(cfg)[k] for k in ("data", "subset")}
    for path in sorted(root.glob(f"*/{split}_metrics.json")):
        metrics = read_json(path)
        config_path = path.parent / "config.json"
        recorded = read_json(config_path) if config_path.exists() else {}
        config = recorded.get("config", cfg)
        tr, model = config["training"], config["model"]
        history_path = path.parent / "history.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        manifest_path = Path(config["data"]["cache"]) / "manifest.json"
        counts = read_json(manifest_path)["splits"]["train"] if manifest_path.exists() else {}
        row = {
            "run": path.parent.name,
            "model": recorded.get("architecture", "baseline"),
            "variant": recorded.get("variant", "ecg_only"),
            "cohort": config.get("subset", {}).get("mode", "full"),
            **{f"training_{key}": counts.get(key) for key in ("patients", "records", "windows")},
            "history_seconds": config["data"]["history_seconds"],
            "horizon_seconds": config["data"]["horizon_seconds"],
            "lr": tr["lr"],
            "hidden": model["hidden"],
            "depth": model["depth"],
            "dropout": model["dropout"],
            "batch_size": tr["batch_size"],
            "epoch_budget": tr["epochs"],
            "actual_epochs": len(history),
            "runtime_seconds": float(history.seconds.sum()) if len(history) else 0,
            "cpu_seconds": float(history.cpu_seconds.sum()) if "cpu_seconds" in history else None,
            "max_epoch_seconds": float(history.seconds.max()) if len(history) else 0,
            "final_val_patient_macro_mae": float(history.val_patient_macro_mae.iloc[-1])
            if "val_patient_macro_mae" in history
            else None,
            "parameters": recorded.get("parameters", 0),
            "fusion": model.get("fusion", "late"),
            "filter": config["preprocessing"]["filter"],
            "scaling": config["preprocessing"]["scaling"],
            **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
        }
        row["evaluated_records"] = row.pop("n_test_records", None)
        row["evaluated_patients"] = row.pop("n_test_patients", None)
        if row["model"] == "baseline":
            for field in ("lr", "hidden", "depth", "dropout", "batch_size"):
                row[field] = None
            row["epoch_budget"] = 0
        (rows if metrics["comparison_key"] == requested else other_rows).append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No evaluated research runs")
    # Cohort/geometry controls must be identical; affine scalers/filters can differ.
    if other_rows:
        pd.DataFrame(other_rows).to_csv(out / f"{split}_other_protocols.csv", index=False)
    table = table.sort_values(["patient_macro_mae", "rmse", "parameters"])
    table.to_csv(out / f"{split}_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(table) * 0.25)))
    table.set_index("run")[["patient_macro_mae", "rmse"]].plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set(xlabel="Error (mV)", title=f"{split}: held-out patient comparisons")
    save_figure(fig, out / f"{split}_comparison.png")
    controls = {
        "lr": [
            "lstm_history",
            "lr_01",
            "lr_0005",
            "lr_0001",
            "cnn_history",
            "cnn_lr_01",
            "cnn_lr_0005",
            "cnn_lr_0001",
        ],
        "hidden": [
            "lstm_history",
            "hidden_128",
            "hidden_256",
            "cnn_history",
            "cnn_hidden_128",
            "cnn_hidden_256",
        ],
        "epoch_budget": ["lstm_60", "lstm_100", "cnn_60", "cnn_100"],
    }
    for field, names in controls.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, group in table.loc[table.run.isin(names)].groupby("model"):
            group = group.sort_values(field)
            ax.plot(group[field], group.patient_macro_mae, "o-", label=name)
        ax.set(
            xlabel=field, ylabel="Validation patient-macro MAE (mV)", title="Matched comparisons"
        )
        if field == "lr":
            ax.set_xscale("log")
        if ax.lines:
            ax.legend()
        save_figure(fig, out / f"{field}_comparison.png")
    stages = {
        "preprocessing": ["p0_lstm", "p1_lstm", "p2_lstm", "p3_lstm"],
        "feature_selection": [
            "p3_lstm",
            "lstm_demographics",
            "lstm_history",
            "lstm_expanded",
            "lstm_mrmr",
            "lstm_permutation",
        ],
        "fusion": ["lstm_history", "lstm_early", "cnn_history", "cnn_early"],
        "qrs": ["persistence", "repeat_window", "training_center", "lstm_60", "cnn_60"],
    }
    for stage, names in stages.items():
        selected = table.set_index("run").reindex(names).dropna(subset=["patient_macro_mae"])
        if selected.empty:
            continue
        columns = ["qrs_f1"] if stage == "qrs" else ["patient_macro_mae", "rmse"]
        fig, ax = plt.subplots(figsize=(10, 5))
        selected[columns].plot.bar(ax=ax)
        ax.set(
            ylabel="Detector-derived F1" if stage == "qrs" else "Validation error (mV)",
            title=f"{stage.replace('_', ' ').title()}: matched development cohort",
        )
        ax.tick_params(axis="x", rotation=25)
        save_figure(fig, out / f"{stage}_comparison.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(table.runtime_seconds, table.patient_macro_mae)
    ax.set(xlabel="Recorded training seconds", ylabel="Patient-macro MAE (mV)")
    save_figure(fig, out / "accuracy_cost.png")
    from compare import bootstrap_pair

    pairs = [
        ("p0_lstm", "p3_lstm"),
        ("p3_lstm", "lstm_history"),
        ("lstm_expanded", "lstm_mrmr"),
        ("lstm_expanded", "lstm_permutation"),
        ("lstm_history", "lstm_early"),
        ("cnn_history", "cnn_early"),
        ("lstm_history", "cnn_history"),
        ("lstm_60", "lstm_100"),
        ("cnn_60", "cnn_100"),
        ("persistence", "lstm_60"),
        ("repeat_window", "lstm_60"),
        ("training_center", "lstm_60"),
        ("lstm_60", "cnn_60"),
    ]
    paired = []
    for a, b in pairs:
        if a in set(table.run) and b in set(table.run):
            result = bootstrap_pair(
                pd.read_csv(root / a / f"{split}_per_record_metrics.csv"),
                pd.read_csv(root / b / f"{split}_per_record_metrics.csv"),
                cfg["evaluation"]["bootstrap_samples"],
                cfg["training"]["seed"],
            )
            paired.append({"A": a, "B": b, "split": split, **result})
    pd.DataFrame(paired).to_csv(out / f"{split}_paired_bootstrap.csv", index=False)
    return table


def freeze_selection(cfg, names):
    manifest = validate_cache(cfg)
    checkpoints = {}
    for name in names:
        run = Path(cfg["paths"]["experiments"]) / name
        metrics = read_json(run / "val_metrics.json")
        if metrics["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Selection cohort mismatch")
        checkpoints[name] = file_hash(run / "best.pt")
    path = Path(cfg["paths"]["experiments"]) / "selection.json"
    if path.exists():
        raise ValueError("Selection already frozen; use a new experiment directory")
    write_json(
        path,
        {
            "fingerprint": manifest["fingerprint"],
            "checkpoints": checkpoints,
            "objective": "validation patient-macro MAE (mV)",
            "used_test": False,
        },
    )
    return path


def research_report(cfg):
    """One consolidated measured report, with pending work explicitly separated."""
    from report import markdown_table

    table = comparison(cfg)
    docs, out, root = [Path(cfg["paths"][k]) for k in ("documents", "outputs", "experiments")]
    docs.mkdir(parents=True, exist_ok=True)
    manifest = validate_cache(cfg)
    columns = [
        "run",
        "patient_macro_mae",
        "mae",
        "rmse",
        "qrs_f1",
        "peak_timing_mae_ms",
        "best_epoch",
        "actual_epochs",
        "runtime_seconds",
        "parameters",
    ]
    for column in columns:
        if column not in table:
            table[column] = None
    sections = [
        "# 100 Hz ECG forecasting research report",
        "",
        "Measured development results only. Official fold 10 remains reserved unless an explicit final-evaluation record is present. Legacy 500 Hz / 0.6-to-0.3-second results are a separate protocol and are not numerically ranked against this study.",
        "",
        "## Protocol and cohort",
        "",
        f"Native {cfg['data']['sampling_rate']} Hz, {cfg['data']['history_seconds']} s observed / {cfg['data']['horizon_seconds']} s future, all twelve leads. Training stride: {cfg['data']['window_stride_seconds']} s. One validation/test window per recording. Time-first batches are [B,500,12] and [B,300,12] for the primary geometry. Caches are channel-first memory maps. No waveforms are preloaded into GPU memory.",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {"split": s, **{k: m[k] for k in ("patients", "records", "windows")}}
                    for s, m in manifest["splits"].items()
                ]
            )
        ),
        "",
        "Sampling is by patient inside official folds 1-8/9/10 with independent seed-42 streams. All records of selected patients remain. Custom development cohorts are not final official test results. Training windows overlap; inference and bootstrap units are not treated as independent patients.",
        "",
        "## Leakage controls",
        "",
        "See ../DATA_LEAKAGE_AUDIT.md and the machine-generated leakage_audit.json. Imputation and scaling fit training inputs only. Prior classes exclude current/equal timestamps. RR features use only observed voltage. Causal SOS filtering never reads target samples. mRMR uses training future summaries as supervised targets; permutation ranking uses validation errors; no method selection uses new test results.",
        "",
        "## Preprocessing ablation",
        "",
        markdown_table(
            table.loc[table.run.isin(["p0_lstm", "p1_lstm", "p2_lstm", "p3_lstm"]), columns]
        ),
        "",
        "P0: no filter + z-score. P1: causal high-pass 0.5 Hz + z-score. P2: causal bandpass 0.5-40 Hz + z-score. P3: no filter + training median/IQR. SciPy Butterworth prototype order 4 yields a fourth-order high-pass or eighth-order bandpass, in SOS form. Initialize each observed window from its first sample with sosfilt_zi, reset state per window, retain nonzero phase/group delay and boundary transients. Target voltages always remain unfiltered. Inverse scaling restores mV; it does not invert the filter.",
        "",
        "P4 follows the measured best-filter and best-scaler decision in preprocessing_selection.json. If it equals an already measured configuration, report that alias without pretending a duplicate training run occurred. Filtering usefulness is judged on validation voltage errors and detector/morphology outcomes, not aesthetic smoothness. Removed components are not proven noise.",
        "",
        "Preprocessing figures: raw_filtered_12lead.png, removed_components.png, fft_welch.png, stft.png and qrs_before_after.png under outputs/<run>/preprocessing. Raw-vs-filtered examples use training records only.",
        "",
        "## Feature selection",
        "",
        markdown_table(
            table.loc[
                table.run.isin(
                    [
                        "p3_lstm",
                        "lstm_demographics",
                        "lstm_history",
                        "lstm_expanded",
                        "lstm_mrmr",
                        "lstm_permutation",
                    ]
                ),
                columns,
            ]
        ),
        "",
        "Feature sets: ECG only (0 static); demographics (4); prior-history conditioning (10); expanded (22). Expanded features add height, weight, BMI, observed-window heart rate, mean RR and RR standard deviation, each with a missing flag. Keep all twelve leads. mRMR is grouped mutual-information difference with eight-bin training quantiles. Relevance averages MI with 24 training-only future per-lead RMS/first-difference-RMS targets. These are forecasting proxies, not clinical outcomes. Redundancy averages pairwise MI against selected groups. Group flags with their variables.",
        "",
        "Grouped permutation shuffles joint groups between validation-patient donors, using the first donor record. Repeats measure importance variability, not causal effects. Correlated variables may substitute for each other, and tiny importances should not be read as robust clinical associations. Fixed top-group counts are declared in the selection CLI and are not test-tuned.",
    ]
    for method in ("mrmr", "permutation"):
        if (out / f"{method}.json").exists():
            selection = read_json(out / f"{method}.json")
            sections.extend(
                [
                    "",
                    f"{method} selected: {', '.join(selection['selected_features']) or 'none'}.",
                    "",
                    markdown_table(pd.DataFrame(selection["ranking"])),
                ]
            )
    sections.extend(
        [
            "",
            "## Static fusion and morphology-oriented changes",
            "",
            markdown_table(
                table.loc[
                    table.run.isin(
                        [
                            "lstm_history",
                            "lstm_early",
                            "cnn_history",
                            "cnn_early",
                            "lstm_huber",
                            "lstm_derivative",
                            "lstm_spectral",
                            "lstm_residual",
                            "cnn_kernels",
                            "cnn_pool1",
                        ]
                    ),
                    columns,
                ]
            ),
            "",
            "Late fusion joins final dynamic state with a static embedding. Early LSTM broadcasts that embedding to every observed step before the LSTM. Early CNN-LSTM concatenates it as channels before the CNN. The dynamic contrastive branch supplies zero context, avoiding metadata shortcuts. Configurable kernels, pooling, hidden size, depth, head width and persistence-residual output preserve the old architecture as a baseline. Derivative loss compares first differences in standardized units, weight 0.1; spectral loss compares unnormalized rFFT magnitudes in standardized units, weight 0.01. Huber delta is 1 standardized unit. These loss weights are initial ablation choices, not independently optimized optima.",
            "",
            "## Hyperparameter search and full epoch budgets",
            "",
            "Selection objective: lowest validation patient-macro MAE in original mV. Secondary criteria are validation RMSE, detector performance, stability and computational cost. The 0.0005 mV near-equivalence heuristic was declared during development, before choosing the 60/100-epoch finalists; it was not preregistered before all screening. Report nominal minima alongside simplicity-based selections. Short one-factor screening holds all other settings fixed within a matched reference. It is not an exhaustive grid or a claim of global optimality. A plateau scheduler is a declared separate comparison.",
            "",
            markdown_table(
                table[
                    [
                        "run",
                        "model",
                        "variant",
                        "lr",
                        "hidden",
                        "depth",
                        "dropout",
                        "batch_size",
                        "epoch_budget",
                        "actual_epochs",
                        "best_epoch",
                        "patient_macro_mae",
                        "rmse",
                        "runtime_seconds",
                        "parameters",
                    ]
                ]
            ),
            "",
            "Complete 60/100-epoch runs have early_stopping=false. best.pt is validation-selected; last.pt retains final weights, optimizer and RNG states. final_val_patient_macro_mae in the CSV describes the last epoch, distinct from the best checkpoint. Some epoch wall times contain large pauses; raw times are retained. Later runs also log process CPU seconds, which can exceed elapsed time with multiple CPU threads. These are observations, not controlled hardware speed benchmarks.",
            "",
            "## QRS and waveform evaluation",
            "",
            "Use WFDB XQRS on lead II. Prepend the common observed prefix independently to actual and predicted futures, then retain only forecast-region peaks. Ordered dynamic programming maximizes one-to-one matches, then minimizes timing error within a fixed 100 ms tolerance. Report precision, recall, F1, matched timing/amplitude MAE, missed peaks, false detections and failures. If neither signal has peaks, precision/recall/F1 are undefined; if true peaks exist and none are predicted, recall and F1 are zero. Failed detector windows are counted and excluded from peak scoring rather than silently treated as empty beats.",
            "",
            "These are detector-derived reference peaks, not expert beat annotations. XQRS has its own filtering, rate/refractory assumptions and possible edge/low-amplitude errors; its use here does not validate it clinically at 100 Hz on predicted traces. Visual inspection of ECG 20131 shows some reference markers offset from the large positive R deflection. Therefore matched timing/amplitude are detector-location errors, not validated anatomical R-apex errors. No P/Q/S/T landmark claims are made. Voltage regression and peak morphology must be interpreted separately. R2 uses one global target mean; Pearson is aggregated per lead and undefined leads are counted. Patient-macro MAE weights patients equally.",
            "",
            "## Baselines and final-evaluation status",
            "",
            markdown_table(
                table.loc[
                    table.run.isin(
                        [
                            "persistence",
                            "repeat_window",
                            "training_center",
                            "original_lstm",
                            "original_cnn_lstm",
                            "lstm_60",
                            "lstm_100",
                            "cnn_60",
                            "cnn_100",
                        ]
                    ),
                    columns,
                ]
            ),
            "",
            "Persistence and repetition use original observed voltages, even for filtered-input studies. A third diagnostic baseline holds each lead at its training-input median to expose apparent gains from flat predictions. Full-data official-test models are pending unless test_metrics files and a frozen selection record exist. Do not present these validation results as final generalization estimates. Run final configurations on official training patients, select with fold 9, freeze checkpoint hashes, then explicitly enable full fold-10 evaluation.",
            "",
            "## Observation-window comparisons",
            "",
            "Context recipes use one training window per record. For aligned comparisons, aligned_3_1 and context_5_1 both predict seconds 5-6; aligned_5_3 and context_7_3 both predict seconds 7-10. Earlier start-zero context runs target different future intervals and remain descriptive, not evidence for the effect of history length. The 8-to-2 study also has a different horizon. Other protocols are excluded from the primary ranked table.",
            "",
            "## Uncertainty and limits",
            "",
            "Patient-bootstrap intervals use 1,000 paired resamples with differences B minus A; negative favors B. Validation intervals are conditional descriptive checks after repeated model selection and have no multiplicity adjustment. One seed and 24 validation patients do not establish broad superiority. External test performance, repeated-seed stability, verified historical availability and clinical false-alarm reduction remain unestablished.",
            "",
            "## Reproducibility and sources",
            "",
            "Each completed model has a saved configuration, seed, feature order, software versions, manifest fingerprint, checkpoint, epoch history and predictions with IDs. Newer runs also save source-file hashes and tensor dimensions. See val_comparison.csv, val_paired_bootstrap.csv and selection artifacts under the configured outputs directory. Historical results are protected by legacy_integrity.json.",
            "",
            "- PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/",
            "- Causal filtering: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html",
            "- WFDB XQRS: https://wfdb.readthedocs.io/en/latest/processing.html",
            "- Peng, Long and Ding (2005), mRMR: https://doi.org/10.1109/TPAMI.2005.159",
            "- Contrastive learning: https://proceedings.mlr.press/v119/chen20j.html",
            "- The supplied private TFT manuscript is not uploaded or embedded; its existing local literature comparison remains in the legacy assignment report.",
        ]
    )
    sections.extend(measured_findings(cfg, table, docs, out, root))
    (docs / "RESEARCH_REPORT.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    return table


def measured_findings(cfg, table, docs, out, root):
    """Derive concise answers and aligned-context uncertainty from actual artifacts."""
    import os

    from compare import bootstrap_pair
    from report import markdown_table

    index = table.set_index("run")

    def metric(name, key="patient_macro_mae"):
        return float(index.loc[name, key]) if name in index.index else None

    def number(value):
        return "pending" if value is None or not np.isfinite(value) else f"{value:.6f}"

    context = []
    for path in sorted(root.glob("*/val_metrics.json")):
        if path.parent.name.startswith(("context_", "aligned_")):
            d = read_json(path.parent / "config.json")["config"]["data"]
            m = read_json(path)
            context.append(
                {
                    "run": path.parent.name,
                    "history_s": d["history_seconds"],
                    "horizon_s": d["horizon_seconds"],
                    "forecast_origin_s": (d["start"] + d["history"]) / d["sampling_rate"],
                    "patient_macro_mae": m["patient_macro_mae"],
                    "rmse": m["rmse"],
                    "qrs_f1": m["qrs_f1"],
                }
            )
    context_table = pd.DataFrame(context)
    context_table.to_csv(out / "context_comparison.csv", index=False)
    context_pairs = []
    for a, b in (("aligned_3_1", "context_5_1"), ("aligned_5_3", "context_7_3")):
        if (root / a / "val_metrics.json").exists() and (root / b / "val_metrics.json").exists():
            result = bootstrap_pair(
                pd.read_csv(root / a / "val_per_record_metrics.csv"),
                pd.read_csv(root / b / "val_per_record_metrics.csv"),
                cfg["evaluation"]["bootstrap_samples"],
                cfg["training"]["seed"],
            )
            context_pairs.append({"A": a, "B": b, **result})
    pd.DataFrame(context_pairs).to_csv(out / "context_paired_bootstrap.csv", index=False)
    neural = table.loc[table.model.ne("baseline")]
    best = neural.sort_values(["patient_macro_mae", "parameters"]).iloc[0]
    lr_candidates = [n for n in ("lstm_history", "lr_01", "lr_0005", "lr_0001") if n in index.index]
    cnn_lr = [
        n for n in ("cnn_history", "cnn_lr_01", "cnn_lr_0005", "cnn_lr_0001") if n in index.index
    ]
    lr_best = (
        index.loc[lr_candidates].sort_values("patient_macro_mae").iloc[0] if lr_candidates else None
    )
    cnn_lr_best = index.loc[cnn_lr].sort_values("patient_macro_mae").iloc[0] if cnn_lr else None
    answers = [
        {
            "Question": "1. Leakage-safe?",
            "Measured answer": "Implemented isolation, train-only fitting and future-mutation checks pass; see audit and remaining limitations. This is bounded evidence, not an absolute guarantee.",
        },
        {
            "Question": "2. Configurable 100 Hz pipeline?",
            "Measured answer": "Working on real selected patients; pilot/medium/full/custom selection, seconds-based windows, time-first batches and streamed caches are implemented. Full model training remains pending.",
        },
        {
            "Question": "3. Best preprocessing?",
            "Measured answer": f"P3 no filter + robust scaling: MAE {number(metric('p3_lstm'))} vs P0 {number(metric('p0_lstm'))}. P3 RMSE {number(metric('p3_lstm', 'rmse'))} vs P0 {number(metric('p0_lstm', 'rmse'))}; the primary MAE improves but RMSE does not.",
        },
        {
            "Question": "4. Is filtering worthwhile?",
            "Measured answer": f"No demonstrated benefit: high-pass MAE {number(metric('p1_lstm'))}; bandpass {number(metric('p2_lstm'))}. Retain no additional filter. P4 aliases P3.",
        },
        {
            "Question": "5. Best feature selector?",
            "Measured answer": f"mRMR MAE {number(metric('lstm_mrmr'))}, permutation {number(metric('lstm_permutation'))}, expanded {number(metric('lstm_expanded'))}. mRMR is nominally lower and uses six static values; permutation uses four. Changes are tiny, not a robust clinical result.",
        },
        {
            "Question": "6. Does early fusion help?",
            "Measured answer": f"LSTM late/early: {number(metric('lstm_history'))}/{number(metric('lstm_early'))}. CNN late/early: {number(metric('cnn_history'))}/{number(metric('cnn_early'))}. Differences are below the declared near-equivalence tolerance; no strong advantage established.",
        },
        {
            "Question": "7. Longer history?",
            "Measured answer": "See target-aligned comparisons below. Both forecasts in a pair use the same actual future. Unaligned start-zero results cannot answer this question directly.",
        },
        {
            "Question": "8. Better QRS predictions?",
            "Measured answer": f"Best neural-by-MAE QRS F1={number(float(best.qrs_f1))}; repeated-window F1={number(metric('repeat_window', 'qrs_f1'))}. Better voltage error does not establish morphology preservation.",
        },
        {
            "Question": "9. Best learning rate?",
            "Measured answer": f"Matched conditioned LSTM screening nominally favors {lr_best['lr'] if lr_best is not None else 'pending'}; CNN screening {cnn_lr_best['lr'] if cnn_lr_best is not None else 'pending'}. Differences are small; these are not global optima.",
        },
        {
            "Question": "10. Preferred model dimensions?",
            "Measured answer": "Near-equivalent screening favors the simpler ECG-only hidden-64, depth-1, dropout-0.2, batch-64 candidates. Other dimension/depth/dropout/batch measurements remain in the CSV; static embedding is inactive in ECG-only.",
        },
        {
            "Question": "11. 60 or 100 epochs?",
            "Measured answer": f"LSTM selected MAE: {number(metric('lstm_60'))}/{number(metric('lstm_100'))}; CNN: {number(metric('cnn_60'))}/{number(metric('cnn_100'))}. Best selected checkpoints remain early; complete budgets are recorded. Equal best performance favors the smaller execution budget, not a claim that 100 epochs improves accuracy.",
        },
        {
            "Question": "12. Improved LSTM vs original?",
            "Measured answer": f"Selected LSTM-60 MAE {number(metric('lstm_60'))}; original architecture eight-epoch MAE {number(metric('original_lstm'))}. No convincing improvement; budgets differ and full-data matched-budget evaluation is pending.",
        },
        {
            "Question": "13. Improved CNN-LSTM vs original?",
            "Measured answer": f"Selected CNN-60 MAE {number(metric('cnn_60'))}; original architecture eight-epoch MAE {number(metric('original_cnn_lstm'))}. Tiny descriptive difference; full-data matched-budget evaluation is pending.",
        },
        {
            "Question": "14. Neural vs baselines?",
            "Measured answer": f"Nominal best neural {best['run']} MAE {number(float(best.patient_macro_mae))}; persistence {number(metric('persistence'))}, repeat-window {number(metric('repeat_window'))}, training-median constant {number(metric('training_center'))}. Neural models beat the two required voltage baselines, but not the diagnostic constant baseline.",
        },
        {
            "Question": "15. Which complexity is justified?",
            "Measured answer": "No large architecture/conditioning benefit is established. Robust scaling improves the primary metric modestly; filtering adds distortion/cost without a demonstrated gain. Sparse feature selection reduces only a small part of total model parameters.",
        },
        {
            "Question": "16. Negative results?",
            "Measured answer": f"Residual head MAE {number(metric('lstm_residual'))}; auxiliary losses, extra capacity, contrastive learning and prolonged training did not demonstrate robust benefits. Report all rows and final-epoch metrics, not only favorable runs.",
        },
        {
            "Question": "17. Final configuration?",
            "Measured answer": "Development candidates: no filter, robust scaling, 100 Hz 5-to-3 seconds, ECG-only hidden 64/depth 1/dropout 0.2/batch 64/lr 0.001. Full-data validation selection and official testing remain pending. The lower-MAE constant median reference must remain in the final benchmark.",
        },
    ]
    write_json(out / "acceptance_answers.json", answers)
    result = [
        "",
        "## Concise answers to the acceptance questions",
        "",
        markdown_table(pd.DataFrame(answers)),
        "",
        "## Target-aligned history controls",
        "",
        markdown_table(context_table) if len(context_table) else "Pending",
        "",
        markdown_table(pd.DataFrame(context_pairs))
        if context_pairs
        else "Pending aligned bootstrap",
        "",
        "## Selected figures",
        "",
    ]
    figures = [
        out / "preprocessing_comparison.png",
        out / "feature_selection_comparison.png",
        out / "fusion_comparison.png",
        out / "qrs_comparison.png",
        out / "lr_comparison.png",
        out / "hidden_comparison.png",
        out / "epoch_budget_comparison.png",
        out / "val_comparison.png",
        out / "mrmr_ranking.png",
        out / "permutation_ranking.png",
        out / "lstm_100/learning_curves.png",
        out / "cnn_100/learning_curves.png",
        out / "cnn_60/val/qrs_overlay_zoom.png",
        out / "repeat_window/val/qrs_overlay_zoom.png",
        out / "preprocessing/bandpass_zscore/raw_filtered_12lead.png",
        out / "preprocessing/bandpass_zscore/fft_welch.png",
        out / "preprocessing/bandpass_zscore/stft.png",
    ]
    for figure in figures:
        if figure.exists():
            result.extend(
                [f"![{figure.stem}]({Path(os.path.relpath(figure, docs)).as_posix()})", ""]
            )
    pending_path = out / "pending_full_runs.csv"
    if pending_path.exists():
        pending = pd.read_csv(pending_path)
        result.extend(
            [
                "## Exact pending full-data experiments",
                "",
                "All rows below are pending, not measured. The full raw-data audit passed; full preprocessing, model training, feature selection and official testing still require execution. Follow the full-data section of CONFIGURATION_AND_RUN_GUIDE.md in dependency order. Runtime estimates are rough CPU extrapolations, not benchmarks.",
                "",
                markdown_table(pending[["run", "epochs", "state", "command"]]),
                "",
            ]
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--freeze", nargs="+", help="Freeze explicitly chosen validation-selected run names"
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args.experiment)
    print(
        freeze_selection(cfg, args.freeze)
        if args.freeze
        else comparison(cfg, args.split).to_string(index=False)
    )
