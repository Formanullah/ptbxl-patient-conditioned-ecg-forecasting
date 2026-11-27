"""Build a report from saved evidence; never train or invent missing results."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare import collect, paired_comparisons
from config import read_json, study_dir, write_json
from visualize import save

SUMMARY_ORDER = [
    "Dataset",
    "Patient-safe folds",
    "Preprocessing",
    "LSTM",
    "CNN-LSTM",
    "Training",
    "RMSE/MAE",
    "Twelve-lead prediction",
    "Evidence-based comparison",
]


def markdown_table(frame):
    if frame.empty:
        return "Pending: no completed comparable research measurements."

    def cell(value):
        if pd.isna(value):
            return "pending"
        return f"{value:.6g}" if isinstance(value, float) else str(value).replace("|", "/")

    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def comparison_figures(cfg, table, paired):
    measured = table.loc[table.state.eq("measured")].copy()
    if measured.empty:
        return
    out = study_dir(cfg) / cfg["run_mode"] / "figures" / "comparisons"
    for split in [s for s in sorted(measured.split.unique()) if s == "test"]:
        selected = measured.loc[measured.split.eq(split)]
        fig, ax = plt.subplots(figsize=(max(9, len(selected) * 0.7), 5))
        selected = selected.assign(
            label=selected.architecture + "/" + selected.preprocessing + "/" + selected.feature_set
        )
        selected.set_index("label")[["mae_mV", "rmse_mV"]].plot.bar(ax=ax)
        ax.set(ylabel="Error (mV)", xlabel="Recorded experiment")
        save(
            fig,
            out / f"{split}_errors.png",
            f"Measured {split} MAE/RMSE are shown for the same waveform task.",
            "Rows with different feature or optimizer settings compare complete recipes; only controlled pairs isolate one factor.",
        )
    for split in ("val", "test"):
        for kind, contrast in (
            ("preprocessing", "P0 versus P1"),
            ("features", "static-feature ablation"),
        ):
            pairs = paired.loc[paired.split.eq(split) & paired.contrast.eq(contrast)]
            if pairs.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.5)))
            values = pairs.difference_B_minus_A_mV.to_numpy()
            errors = [
                np.maximum(0, values - pairs.ci95_low.to_numpy()),
                np.maximum(0, pairs.ci95_high.to_numpy() - values),
            ]
            labels = [f"{a} → {b}" for a, b in zip(pairs.A, pairs.B, strict=True)]
            ax.barh(np.arange(len(pairs)), values, xerr=errors)
            ax.set(
                yticks=np.arange(len(pairs)),
                yticklabels=labels,
                xlabel="B − A patient-macro MAE (mV)",
            )
            ax.axvline(0, color="black", lw=0.8)
            save(
                fig,
                out / f"{kind}_{split}.png",
                f"Matched {contrast} differences and paired patient-bootstrap intervals are shown.",
                "Negative differences favor B; intervals crossing zero are inconclusive, and single-seed intervals do not capture training variability.",
            )
    lr = measured.loc[measured.split.eq("val") & measured.lr.notna()]
    if len(lr):
        fig, ax = plt.subplots(figsize=(10, 5))
        for (model, filtering, features, hidden), group in lr.groupby(
            ["architecture", "preprocessing", "feature_set", "hidden"]
        ):
            if group.lr.nunique() > 1:
                group = group.sort_values("lr")
                ax.plot(
                    group.lr,
                    group.patient_macro_mae_mV,
                    marker="o",
                    label=f"{model}/{filtering}/{features}/h{hidden:g}",
                )
        if ax.lines:
            ax.set(xscale="log", xlabel="Learning rate", ylabel="Validation patient-macro MAE (mV)")
            ax.legend(fontsize=7)
            save(
                fig,
                out / "learning_rates.png",
                "Validation results compare the tested learning rates within fixed recipes.",
                "The selected rate is best among these measured candidates, not a proven global optimum.",
            )
        else:
            plt.close(fig)


def generate(cfg):
    root = study_dir(cfg) / cfg["run_mode"]
    root.mkdir(parents=True, exist_ok=True)
    table, leads, records = collect(cfg)
    paired = paired_comparisons(cfg, table, records)
    table.to_csv(root / "experiment_results.csv", index=False)
    leads.to_csv(root / "per_lead_metrics.csv", index=False)
    paired.to_csv(root / "paired_comparisons.csv", index=False)
    comparison_figures(cfg, table, paired)
    manifest_path = root / "cohort.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    counts = pd.DataFrame(manifest["splits"]).T if manifest else pd.DataFrame()
    measured = table.loc[table.state.eq("measured")]
    smoke = (
        table.loc[
            table.state.eq("software smoke check passed; excluded from research comparison")
            & table["synthetic"].eq(False)
        ]
        if "synthetic" in table
        else table.iloc[:0]
    )
    metrics_columns = [
        "name",
        "split",
        "mae_mV",
        "rmse_mV",
        "patient_macro_mae_mV",
        "mse_mV2",
        "r2",
        "pearson_mean",
    ]
    hardware = [read_json(p)["environment"] for p in (root / "checkpoints").glob("*/run.json")]
    model_rows = []
    for path in (root / "checkpoints").glob("*/run.json"):
        run = read_json(path)
        model_rows.append(
            {
                "model": run["architecture"],
                "filter": run["preprocessing"],
                "features": run["feature_set"],
                "static_values": len(run["selected_features"]),
                "hidden": run["config"]["model"]["hidden"],
                "layers": run["config"]["model"]["depth"],
                "active_parameters": run["parameters"],
                "output": f"[B,{cfg['data']['horizon']},12]",
            }
        )
    model_table = pd.DataFrame(
        model_rows,
        columns=[
            "model",
            "filter",
            "features",
            "static_values",
            "hidden",
            "layers",
            "active_parameters",
            "output",
        ],
    )
    status_columns = ["name", "state", "epochs_completed", "best_epoch", "runtime_seconds"]
    sections = {
        "Dataset": f"PTB-XL v1.0.3; source: https://physionet.org/content/ptb-xl/1.0.3/. Data root: `{cfg['data']['root']}/records100`. WFDB reads physical mV, 100 Hz, twelve leads, 1,000 samples per record. History/target: `{cfg['data']['history']}/{cfg['data']['horizon']}`; batched input `[B,{cfg['data']['history']},12]`, output `[B,{cfg['data']['horizon']},12]`. This is within-record waveform forecasting, not clinical longitudinal prediction.",
        "Patient-safe folds": "Official folds 1–8 train, 9 validate, 10 test; patients remain disjoint. Selected whole patients retain all eligible ECGs.\n\n"
        + markdown_table(counts.reset_index(names="split"))
        + "\n\nReserved fold 10 metadata counts: "
        + (
            str({k: v for k, v in manifest["reserved_test"].items() if k != "prepared"})
            if manifest
            else "preparation pending"
        )
        + "\n\nFinal-test ledger: "
        + (
            str(read_json(root / "test_ledger.json")["conditions"])
            if (root / "test_ledger.json").exists()
            else "pending / not enabled"
        ),
        "Preprocessing": "P0 uses raw history; P1 uses causal 0.5–40 Hz SOS Butterworth filtering on observed history only (prototype N=4; bandpass order 8). Both use identical raw-training-history z-scores and predict the same original-mV future. Filtering may distort ST segments, QRS shape, phase, and initial samples; it is a hypothesis. History-only distortion/PSD figures do not establish physiological accuracy.",
        "LSTM": "Independent `LSTMForecaster`: optional early static embedding → stacked LSTM → final hidden state → MLP → all future samples. Exact active parameter counts and model settings are in each run.json; see docs/EQUATIONS.md.",
        "CNN-LSTM": "Independent `CNNLSTMForecaster`: optional static channels → Conv1d(7,32) → ReLU → Conv1d(5,64) → ReLU → AvgPool(2) → LSTM → MLP. The primary 700-step history becomes 350 recurrent steps. No attention or contrastive projector is active.",
        "Training": "AdamW, normalized MSE, fixed learning rates, gradient clipping, seed 42; validation patient-macro MAE selects checkpoints. No early stopping truncates the budget.\n\n"
        + markdown_table(table.reindex(columns=status_columns))
        + "\n\nRecorded environments: `"
        + str(hardware)
        + "`. Full 120-epoch Colab runs remain pending unless their completed artifacts are present.",
        "RMSE/MAE": "MAE = mean(|prediction − truth|), in mV; RMSE = sqrt(mean((prediction − truth)²)), in mV; MSE is mV². RMSE applies to both regression architectures and is not an LSTM gate. Patient-macro MAE averages records within each patient, then weights patients equally.\n\n"
        + markdown_table(measured.reindex(columns=metrics_columns)),
        "Twelve-lead prediction": "Saved figures show each lead separately with observed history, original future truth, forecast, physical axes, and the forecast boundary. The pre-specified example is selected by metadata ID before errors; the median-error example is labelled. No fabricated peak markers or unvalidated P/Q/R/S/T delineation are used.\n\n"
        + markdown_table(leads.head(24)),
        "Evidence-based comparison": markdown_table(paired)
        + "\n\nNegative B−A favors B. Intervals resample patients and are descriptive after validation selection, using one seed without multiplicity correction. Medium/full cohorts and different horizons are not pooled. Low voltage error does not establish clinical validity; 100-Hz resolution and filter/morphology limitations remain. ARIMA is deferred. Feature-selection execution is pending unless feature_extension.json exists. Synthetic and pilot runs are software checks, not research performance evidence.",
    }
    for model, heading in (("lstm", "LSTM"), ("cnn_lstm", "CNN-LSTM")):
        sections[heading] += "\n\n" + markdown_table(
            model_table.loc[model_table.model.eq(model)].drop_duplicates()
        )
    if not smoke.empty:
        sections["RMSE/MAE"] += (
            "\n\nSoftware pilot diagnostics (excluded from research rankings):\n\n"
            + markdown_table(smoke.reindex(columns=metrics_columns))
        )
    filter_comparisons = paired.loc[paired.split.eq("val") & paired.contrast.eq("P0 versus P1")]
    sections["Preprocessing"] += "\n\nMeasured validation filter effects:\n\n" + markdown_table(
        filter_comparisons
    )
    milestone_rows = []
    for path in (root / "metrics").glob("*_through_*/val.json"):
        item = read_json(path)
        if not item["smoke"]:
            milestone_rows.append(
                {k: item[k] for k in ("name", "best_epoch", "patient_macro_mae_mV", "rmse_mV")}
            )
    sections["Training"] += "\n\n60/120 checkpoint evidence:\n\n" + markdown_table(
        pd.DataFrame(milestone_rows)
    )
    if manifest and manifest["synthetic"]:
        warning = "**SYNTHETIC SOFTWARE FIXTURE — no PTB-XL performance claims.**\n\n"
    elif cfg["run_mode"] == "pilot":
        warning = "**REAL-DATA SOFTWARE PILOT — numerical diagnostics are excluded from research rankings.**\n\n"
    else:
        warning = ""
    text = (
        "# PTB-XL research results\n\n"
        + warning
        + "\n\n".join(f"## {name}\n\n{sections[name]}" for name in SUMMARY_ORDER)
    )
    (root / "RESULTS.md").write_text(text + "\n", encoding="utf-8")
    write_json(
        root / "summary.json", {"order": SUMMARY_ORDER, "sections": sections, "warning": warning}
    )
    docs = Path("docs")
    if docs.exists() and manifest and not manifest["synthetic"]:
        (docs / "RESULTS.md").write_text(text + "\n", encoding="utf-8")
    return root / "RESULTS.md"


def bundle(cfg):
    root = study_dir(cfg)
    destination = root / "PTBXL_Research_Results.zip"
    temporary = destination.with_suffix(".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix not in (".zip", ".tmp", ".lock"):
                archive.write(path, "results/" + path.relative_to(root).as_posix())
        for path in Path("docs").glob("*.md"):
            archive.write(path, path.as_posix())
        archive.write("configs/config.yaml", "configs/config.yaml")
    temporary.replace(destination)
    return destination
