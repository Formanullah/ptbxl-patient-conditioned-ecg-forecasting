"""Build the assignment report from the selected configuration and saved results.

Reporting never trains a model, prepares data, or runs model inference.
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from compare import compare_runs, current_manifest
from config import load_config, read_json, write_json


def markdown_table(frame):
    """Render a small table without an optional Markdown dependency."""
    if frame.empty:
        return "No completed measurements for this configuration."

    def cell(value):
        if value is None or (np.isscalar(value) and pd.isna(value)):
            return "undefined"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    rows.append("| " + " | ".join("---" for _ in frame.columns) + " |")
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def relative_link(path, document_dir):
    return Path(os.path.relpath(path, document_dir)).as_posix()


def requested_cohort(cfg):
    subset = cfg["subset"]
    if subset["mode"] == "full":
        return "All patients in official folds 1–8 / 9 / 10."
    values = subset.get("fractions", subset.get("patients", {}))
    unit = "fraction" if "fractions" in subset else "patients"
    return ", ".join(f"{split}: {values.get(split, 'preset')} {unit}" for split in ("train", "val", "test"))


def measured_findings(table, split):
    """Describe only scores that exist for the current cohort."""
    if table.empty:
        return ["No model-performance conclusion is available for this cohort yet."]
    neural = table.loc[table.model.ne("baseline")]
    baselines = table.loc[table.model.eq("baseline")]
    if neural.empty:
        return ["Only baseline evaluations are available. Neural-model evaluation is pending."]
    if split == "test":
        return ["These are final descriptive test measurements. Do not use them to retune or select models."]
    best = neural.sort_values(["patient_macro_mae", "parameters"]).iloc[0]
    findings = [
        f"Lowest measured neural validation patient-macro MAE: **{best['run']}**, "
        f"**{best.patient_macro_mae:.6f} mV**. This is a descriptive validation minimum, not a global optimum."
    ]
    if not baselines.empty:
        for row in baselines.itertuples(index=False):
            difference = best.patient_macro_mae - row.patient_macro_mae
            direction = "lower" if difference < 0 else "higher"
            findings.append(
                f"Its MAE is {abs(difference):.6f} mV {direction} than {row.run} "
                f"({row.patient_macro_mae:.6f} mV)."
            )
    if pd.notna(best.get("qrs_f1")) and best.qrs_f1 == 0:
        findings.append(
            "The lowest-MAE neural checkpoint has detector-based QRS F1 of zero. "
            "Lower voltage error does not demonstrate useful heartbeat prediction; inspect the waveform overlays."
        )
    for model, group in neural.groupby("model"):
        if len(group) > 1 and group.epoch_budget.nunique() > 1:
            findings.append(
                f"{model} has multiple measured training budgets. Compare actual_epochs, best_epoch and "
                "final_val_patient_macro_mae; a larger budget does not by itself prove improvement."
            )
    return findings


def report(cfg, split="val"):
    """Refresh one active report; changing cohort settings cannot import older scores."""
    if split not in ("val", "test"):
        raise ValueError("Report split must be val or test")
    manifest = current_manifest(cfg)
    table, states = compare_runs(cfg, split)
    out = Path(cfg["paths"]["outputs"])
    docs = Path(cfg["paths"]["documents"])
    docs.mkdir(parents=True, exist_ok=True)
    destination = docs / "ASSIGNMENT_REPORT.md"
    write_json(out / "report_config.json", cfg)
    data = cfg["data"]
    sections = [
        "# Patient-conditioned ECG forecasting: assignment report",
        "",
        f"Generated from saved artifacts at {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC).",
        "",
        "## Current configuration and execution state",
        "",
        f"Requested cohort: **{requested_cohort(cfg)}**",
        "",
        f"Protocol: **{data['sampling_rate']} Hz**, **{data['history_seconds']:g} seconds observed → "
        f"{data['horizon_seconds']:g} seconds forecast**, twelve leads. "
        f"Input/target lengths: **{data['history']} / {data['horizon']} samples**; "
        f"layout: `{data['layout']}`.",
        "",
        f"Preprocessing: `{cfg['preprocessing']['filter']}` filter; `{cfg['preprocessing']['scaling']}` scaling. "
        f"Seed: {cfg['training']['seed']}. Configured maximum: {cfg['training']['epochs']} epochs. "
        f"Early stopping: {cfg['training']['early_stopping']}.",
        "",
        f"Cache: `{data['cache']}`. Experiments: `{cfg['paths']['experiments']}`. "
        f"[Exact report configuration]({relative_link(out / 'report_config.json', docs)}).",
        "",
    ]
    if manifest is None:
        sections.extend([
            "**Preprocessing pending.** Requested patient counts are settings, not measured dataset counts.", ""
        ])
    else:
        counts = pd.DataFrame([{"split": name, **{key: values[key] for key in ("patients", "records", "windows")}}
                               for name, values in manifest["splits"].items()])
        sections.extend([
            "**Data prepared.** Actual cohort counts:", "", markdown_table(counts), "",
            f"Manifest fingerprint: `{manifest['fingerprint']}`. "
            f"Preparation took {manifest['preprocessing_seconds']:.1f} recorded seconds.", ""
        ])
    sections.extend([
        f"Completed comparable **{split} evaluations: {len(table)}**. "
        "No measurements from a different cohort/window protocol are inserted into this report.", "",
        markdown_table(states) if len(states) else "Model training and evaluation have not been recorded for this cohort.", "",
        "## Methods and leakage controls", "",
        "Official folds 1–8 train, fold 9 validates, and fold 10 is reserved for explicit final evaluation. "
        "Selection is by whole patient, retaining their eligible recordings. Training alone is shuffled. "
        "Training can use overlapping windows; validation/test use one configured window per recording.", "",
        "Waveform scaling, age imputation and expanded-feature transforms fit training inputs only. "
        "The age-300 sentinel is missing. Sex has an unknown encoding and a missingness flag. "
        "Historical diagnoses use strictly earlier timestamps; current/equal-time labels are excluded. "
        "Heart-rate and RR features see observed samples only. Causal filters operate on the observed window; "
        "future targets remain unfiltered physical voltages, affine-scaled for training.", "",
        "Feature choices: ECG-only (0 static values), demographics (4), prior history (10), expanded (22), "
        "or groups chosen by training-only mRMR / validation grouped permutation importance. Missingness flags "
        "remain with their features. All models retain twelve ECG leads.", "",
        "LSTM summarizes the observed sequence with recurrent gates. CNN-LSTM first applies three temporal "
        "convolutions and average pooling, then an LSTM. Late fusion joins the temporal state and static "
        "embedding at the forecast head; early fusion broadcasts the embedding before the temporal encoder. "
        "Contrastive learning uses observed-only augmented views and excludes same-patient negatives.", "",
        "Training uses AdamW and gradient clipping. Checkpoints are selected using validation patient-macro "
        "MAE in mV. The supplied settings are practical development choices, not proven optima for every cohort. "
        "best.pt and last.pt have different purposes: selected inference weights versus resumable final state.", "",
        "## Metrics and measured comparison", "",
        "MAE/RMSE are mV; MSE is mV². Patient-macro MAE averages within patients, then equally across patients. "
        "R² uses one global target mean. Pearson is computed per lead; undefined values remain null. "
        "QRS scores compare WFDB XQRS detections on lead II with one-to-one matching within the configured "
        "tolerance. They are detector proxies, not expert annotations; timing/amplitude results can be affected "
        "by misplaced detections. Clinical usefulness is not established by voltage error.", "",
    ])
    metric_columns = ["run", "model", "variant", "patient_macro_mae", "mae", "rmse", "mse", "r2",
                      "pearson_mean", "qrs_f1", "peak_timing_mae_ms", "parameters"]
    sections.extend([markdown_table(table[[c for c in metric_columns if c in table]]), ""])
    if len(table):
        settings = ["run", "filter", "scaling", "fusion", "hidden", "depth", "dropout", "batch_size", "lr",
                    "epoch_budget", "actual_epochs", "best_epoch", "final_val_patient_macro_mae", "runtime_seconds"]
        sections.extend(["## Executed settings and budgets", "", markdown_table(table[settings]), ""])
        sections.extend(["## Findings from this cohort", ""])
        sections.extend(f"- {finding}" for finding in measured_findings(table, split))
        sections.append("")
        pairs = pd.read_csv(out / f"{split}_paired_bootstrap.csv")
        sections.extend([
            "## Paired uncertainty", "",
            "Paired bootstrap resamples patients; differences are B minus A. Negative favors B. "
            "These are descriptive intervals after validation selection, with one seed and no multiplicity correction. "
            "Check the saved run settings before interpreting a difference as an isolated architecture/feature effect.", "",
            markdown_table(pairs), "", "## Figures and detailed artifacts", "",
            f"![Current cohort comparison]({relative_link(out / f'{split}_comparison.png', docs)})", ""
        ])
        for name in table.run:
            run_dir = out / name
            sections.append(f"- **{name}**: [evaluation artifacts]({relative_link(run_dir / split, docs)}).")
        neural = table.loc[table.model.ne("baseline")]
        representatives = neural.groupby("model", sort=True).head(1) if split == "val" else neural.sort_values("run").groupby("model").head(1)
        for name in representatives.run:
            for figure in (out / name / "learning_curves.png", out / name / split / "forecast_median_error.png"):
                if figure.exists():
                    sections.extend(["", f"![{name}: {figure.stem}]({relative_link(figure, docs)})", ""])
    else:
        sections.extend(["## Findings", "", "Training/evaluation pending; no performance values are invented.", ""])
    if manifest:
        for method in ("mrmr", "permutation"):
            path = out / f"{method}.json"
            if path.exists():
                selected = read_json(path)
                if selected["fingerprint"] == manifest["fingerprint"]:
                    sections.extend([f"## {method} selected features", "", ", ".join(selected["selected_features"]), ""])
    sections.extend([
        "## Limitations and next actions", "",
        "Inspect waveform and QRS plots alongside scalar errors. One seed and repeated validation choices "
        "limit conclusions. Overlapping training windows are correlated. Prior-record labels are a proxy for "
        "clinically available history. Historical 500 Hz/short-horizon results and the earlier tiny 100 Hz "
        "development cohort use different protocols/populations and are not copied into this comparison.", "",
        "After changing dataset settings, prepare and audit that cohort, train each desired model, evaluate "
        "on validation, and regenerate this report. Reporting refreshes only when explicitly requested. "
        "For final testing, use full mode, select with full validation, freeze checkpoint hashes and explicitly "
        "enable test access. See [the run guide](CONFIGURATION_AND_RUN_GUIDE.md).", "",
        "## References", "",
        "- [PTB-XL v1.0.3 and official fold guidance](https://physionet.org/content/ptb-xl/1.0.3/)",
        "- [SciPy causal SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html)",
        "- [WFDB processing and XQRS](https://wfdb.readthedocs.io/en/latest/processing.html)",
        "- [Peng et al., mRMR (2005)](https://doi.org/10.1109/TPAMI.2005.159)",
        "- Private-manuscript literature values remain in the archived original assignment report. "
        "Different splits, scaling, sampling rates and horizons prevent direct numerical superiority claims.", ""
    ])
    destination.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report: {destination}; {len(table)} completed {split} evaluations for this cohort")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    report(load_config(args.config, args.experiment), args.split)
