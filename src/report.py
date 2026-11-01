"""Generate measured assignment documentation and an explicit Colab allowlist bundle."""

import argparse
import os
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from compare import compare_runs
from config import VARIANTS, load_config, read_json
from evaluate import save_figure


def metadata_figures(cfg):
    import matplotlib.pyplot as plt

    frames = {
        s: pd.read_csv(Path(cfg["data"]["cache"]) / f"metadata_{s}.csv")
        for s in ("train", "val", "test")
    }
    out = Path(cfg["paths"]["outputs"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for split, frame in frames.items():
        axes[0].hist(frame.age.dropna(), bins=20, alpha=0.4, label=split)
    axes[0].set(xlabel="Age (years)", ylabel="Recordings", title="Observed ages")
    axes[0].legend()
    axes[1].bar(list(frames), [len(f) for f in frames.values()])
    axes[1].set(title="Patient-isolated folds", ylabel="Recordings")
    axes[2].bar(list(frames), [100 * f.history_available.mean() for f in frames.values()])
    axes[2].set(title="Strictly prior history coverage", ylabel="Percent of recordings")
    save_figure(fig, out / "metadata_distributions.png")


def markdown_table(frame):
    def cell(value):
        if pd.isna(value):
            return "undefined"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "/")

    headings = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([headings, rule, *rows])


METHODS = """## Research question and interpretation

Does known baseline context improve short-horizon multivariate ECG forecasting, and
does an auxiliary contrastive objective improve the learned dynamic representation?
The experiment uses age, sex and prior diagnostic superclasses. Demographic or
pathology conditioning is a hypothesis, not an assurance of improvement.

This is within-record voltage forecasting, not prediction of disease at a later visit.
No clinical false-alarm reduction or diagnostic accuracy is established by these errors.

## Data and preprocessing

PTB-XL v1.0.3: 21,799 ECGs, 18,869 patients, twelve leads. The default experiment
uses records500 at 500 Hz: samples 0:300 (0.6 s) predict 300:450 (0.3 s).
Each record contributes one fixed window. Official patient-isolated folds 1-8/9/10
yield 17,418/2,183/2,198 records and 15,023/1,942/1,904 patients.
Waveforms retain physical mV units at ingestion. No additional filtering or resampling
is applied. Standardization fits per-lead population mean/std on training INPUTS only;
both inputs and targets use that transform. Predictions are inverted to mV for metrics.

Age outside 0-120, including sentinel 300, is missing. Training-only median imputation
and population standard deviation scaling are used. Age and sex have missingness
flags; sex follows PTB-XL's male=0/female=1 coding with unknown=0.5 plus its flag.
Diagnostic SCP keys map to NORM, MI, STTC, CD and HYP. Likelihood zero means unknown
confidence, not absence. Historical indicators are the union of labels from strictly
earlier timestamps of the same patient; current/same-time labels are excluded.
Missing history is a separate flag. Available history covers 2,929 records (13.4%);
only 293 test records have it. Historical labels are a proxy, not a verified clinical
problem list. Current-record labels are used only for subgroup reporting.

## Models and hyperparameter justification

LSTM: two layers, hidden size 128; recurrent dropout 0.2. CNN-LSTM adds three padded
temporal convolutions: 12->32 (kernel 7), 32->64 (kernel 5), average pooling by 2,
64->64 (kernel 3), with ReLU. A 300-sample sequence becomes 150 CNN feature steps.
The CNN receptive field after the final convolution is 16 raw samples (32 ms at 500 Hz).
The static branch projects 0/4/10 selected features to 32 dimensions when present.
Final dynamic state and static context enter a 256-unit ReLU/dropout head and 1,800
outputs reshaped to 12x150. Every model owns a 128->128->64 projection head; it is
trained and used only for contrastive variants. Parameter counts include that head;
parameters_used_by_loss in each run distinguishes active training parameters.

Hidden 128 and two layers provide moderate capacity without an extensive sweep.
The 32-dimensional static embedding limits static-branch size. Kernels 7/5/3 capture
local voltage changes; pooling halves recurrent sequence length, with a possible cost
in timing precision. These choices are plausible starting values, not tuned optima.

AdamW learning rate 0.001 is a common initial scale; weight decay 0.0001 and dropout
0.2 regularize. Batch 64 balances memory and minibatch statistics. Clip gradient norm
at 1.0 to limit exploding gradients. Maximum 25 epochs, patience 5, and minimum
validation forecast MSE determine the checkpoint. No test-based tuning is performed.
All eight runs use seed 42, the same splits, cache, optimizer settings and output window.

For contrastive variants, two independent views augment observed inputs only, using
normalized Gaussian jitter std 0.01 and a global gain in [0.95,1.05]. No timing shifts,
polarity reversal or target augmentation occurs. The dynamic encoder supplies an
L2-normalized 64-dimensional projection. Symmetric NT-Xent uses temperature 0.1;
the paired view is positive, self and other same-patient comparisons are excluded.
Anchors without negatives are skipped and counted. The loss is normalized forecast
MSE on clean inputs + 0.1*NT-Xent. Projection happens before static fusion to limit
metadata shortcuts. Temperature and weight are declared defaults, not optimized values.

## Metric definitions

For residual e=prediction-target over records, leads and forecast samples:
MAE=mean(abs(e)) [mV], MSE=mean(e^2) [mV^2], RMSE=sqrt(MSE) [mV]. Overall
R2=1-SSE/SST with one global target mean. Pearson is calculated per lead after
concatenating records/time, then averaged over defined leads. Constant-signal cases
are null/undefined and counted. Patient-macro MAE first averages records per patient,
then patients equally. These amplitude metrics do not establish clinical morphology fidelity.

Age groups: <40, 40-59, 60-79, >=80, unknown. Sex and history availability are separate
strata. Diagnostic groups overlap because labels are multi-label. Subgroup errors
are record-weighted, with record/patient counts. Paired patient-bootstrap comparisons
average record MAE within patients, resample patients 1,000 times, and report percentile
95% intervals for B-A; negative favors B. They measure test-patient uncertainty conditional
on this fitted seed and are descriptive, without multiple-comparison correction.
"""

LITERATURE = """## Literature comparison

The supplied local reference is F. S. Butt et al., *Temporal Fusion Transformers for
Forecasting ECG Signals* (user-supplied 2025 manuscript), pp. 5-6, Table 1.
The PDF is not included in this repository, report bundle or Colab transfer.

| Paper experiment | MAE as reported | RMSE as reported | SMAPE as reported |
| --- | --- | --- | --- |
| 1 | 0.0604 | 0.1234 | 0.9950 |
| 2 | 0.0537 | 0.1184 | 0.9333 |
| 3 | 0.0504 | 0.1135 | 0.8791 |
| 4 | 0.0725 | 0.1466 | 1.0104 |
| 5 | 0.0674 | 0.1402 | 0.9620 |

These are contextual published values, not measured project results. The paper's abstract
states RMSE 0.061 and MAE 0.128, conflicting with Table 1; this report uses the explicit
table and does not silently correct the abstract. MAE exceeding RMSE would be inconsistent
for the same residual set under their standard definitions.

The paper reports 300 historic/150 forecast samples for experiments 1-3, age/sex static
features, robust TorchNormalizer scaling and an 80/10/10 split. Its description of
1000-to-500 Hz downsampling conflicts with PTB-XL's documented native 500 Hz/available
100 Hz recordings. Patient isolation, exact scaling/inverse scaling, lead aggregation,
and some other experiment geometries are insufficiently specified for exact reproduction.
This project uses an explicit 500 Hz protocol, train-input-only z-scores, official patient
folds, prior diagnoses and an auxiliary contrastive objective. Report our original-mV
MAE/RMSE beside those numbers with these caveats; no numerical superiority claim is justified.
SMAPE is omitted from our primary metrics because ECG voltages cross zero.

Wagner et al. (2020) defines PTB-XL, not a directly matching waveform forecast benchmark.
Strodthoff et al. (2021) reports ECG classification; classification AUC is not forecast error.
Chen et al. (2020) motivates two-view contrastive learning; the ECG adaptation here is an
experiment, not a claim that image-domain results transfer to ECG.

Sources:
- PTB-XL v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- Wagner et al.: https://doi.org/10.1038/s41597-020-0495-6
- Strodthoff et al.: https://doi.org/10.1109/JBHI.2020.3022989
- Chen et al.: https://proceedings.mlr.press/v119/chen20j.html
- Butt et al.: local user-provided manuscript; no external upload or redistribution.

## Limitations

The 0.6 s context can contain less than one beat; all results concern a 0.3 s within-record
forecast. Fixed start-zero windows cover only a small part of each ten-second recording.
One seed does not quantify optimization variability. Only a minority has prior ECG history;
historical diagnoses are annotations, not independently verified prior clinical knowledge.
Subgroups can be small and overlapping. Standardization and missingness conventions are
explicit but differ from the reference paper. MSE can produce smooth forecasts with weak
QRS peaks; inspect timing and shape visually. No diagnostic, distant-future or clinical
false-alarm claims follow from these experiments.
"""


def presentation_notes(table):
    return (
        """# Class presentation and speaker notes

The results table contains completed full-data runs only. Check the assignment report
and run src/status.py with the matching configuration to check unfinished models
before presenting final comparisons. Progress files are stored with experiment logs.

1. **Question and task:** forecast twelve cardiac voltage streams from 0.6 s observed to
   the next 0.3 s. Explain why patient context might help; improvement is an empirical question.
2. **Dataset and data integrity:** PTB-XL v1.0.3, 500 Hz, mV, official patient folds. Show
   the audit and train-only preprocessing. Explain why random window splitting leaks patients.
3. **Conditioning:** age/sex/masks -> 4 features; plus five prior diagnosis indicators and
   history mask -> 10. Show strict chronological construction and limited history coverage.
4. **LSTM shapes:** [B,12,300] -> transpose [B,300,12] -> two-layer LSTM -> final [B,128].
   Static [B,10] -> [B,32]; concatenate -> [B,160] -> 256 -> 1800 -> [B,12,150].
5. **LSTM gates:** z_t=[h_(t-1),x_t]. Forget f_t=sigmoid(W_f z_t+b_f), input
   i_t=sigmoid(W_i z_t+b_i), candidate g_t=tanh(W_g z_t+b_g), memory
   c_t=f_t*c_(t-1)+i_t*g_t, output o_t=sigmoid(W_o z_t+b_o), h_t=o_t*tanh(c_t).
   Sigmoid gates control flow in [0,1]; tanh supplies signed content; * is elementwise.
6. **Memory and BPTT:** unroll across 300 samples and backpropagate through both layers.
   The direct memory path contains products of forget gates; it helps but does not eliminate
   vanishing gradients. Exploding gradients motivate norm clipping at 1.0.
7. **CNN operation:** Y[o,t]=b[o]+sum_c sum_k W[o,c,k] X[c,t+k-padding]. PyTorch uses
   cross-correlation. Filters span input leads and local time, share weights across time,
   and learn morphology. Explain padding, stride, channels, ReLU and average pooling.
8. **CNN-LSTM shapes:** [B,12,300] -> conv [B,32,300] -> [B,64,300] -> pool
   [B,64,150] -> conv [B,64,150] -> transpose [B,150,64] -> LSTM [B,128].
   The forecast/static head is shared in design with LSTM; recurrent input size differs.
9. **Contrastive learning:** augment only observed voltage; clean input drives forecast.
   Two views -> shared dynamic encoder -> projection -> normalized vectors. Pull paired
   views together, distinguish different patients, and mask same-patient false negatives.
   Static covariates do not enter the contrastive representation.
10. **Loss and optimization:** MSE + 0.1 NT-Xent (temperature 0.1); AdamW lr 0.001,
    weight decay 0.0001, batch 64, dropout 0.2. Show zero_grad -> forward -> loss ->
    backward -> clip -> optimizer.step. Validation uses forecast MSE without augmentation.
11. **Experiment design:** two architectures x four variants = eight runs, same seed/folds,
    plus persistence/repeated-window baselines. Best validation epoch, patience five,
    maximum 25. Explain why comparing total contrastive loss with plain MSE is misleading.
12. **Results:** insert the table below, learning curves and paired-patient intervals.
    Explain mV/mV^2, MAE/RMSE, R2 and correlation. Negative R2 is possible and meaningful.
13. **Forecasts:** show lowest-ID, median-error and worst-error twelve-lead plots. The
    vertical boundary is forecast origin. Discuss attenuated peaks and timing errors.
14. **Patient strata:** show age/sex/diagnosis/history panels with counts. Diagnostic groups
    overlap; a small group is not evidence of a stable clinical difference.
15. **Literature:** explain Table 1 versus abstract conflicts and protocol/scaling differences.
    Do not rank unaligned numerical errors. The paper remains local.
16. **Conclusions:** state only measured changes. One seed, sparse history and short context
    limit conclusions. Future work: longer contexts, repeated seeds, verified history and
    clinical evaluation before claims about false alarms.

## Architecture diagram

```mermaid
flowchart LR
    X[Observed 12-lead ECG] --> C[Optional temporal CNN]
    C --> L[Two-layer LSTM]
    L --> H[Dynamic state 128]
    S[Age / sex / prior diagnoses] --> E[Static embedding 32]
    H --> F[Concatenate and forecast MLP]
    E --> F
    F --> Y[12 x 150 future voltages]
    H --> P[Projection 64: contrastive training only]
```

## Measured test results

"""
        + markdown_table(table[["run", "mae", "rmse", "patient_macro_mae", "best_epoch"]])
        + "\n"
    )


def report(cfg, config_path=None):
    table = compare_runs(cfg)
    metadata_figures(cfg)
    root = Path(cfg["paths"]["experiments"])
    docs = Path(cfg["paths"]["documents"])
    docs.mkdir(parents=True, exist_ok=True)
    pilot = bool(cfg.get("subset"))

    def asset(name):
        return Path(os.path.relpath(Path(cfg["paths"]["outputs"]) / name, docs)).as_posix()

    expected = {f"{a}_{v}" for a in ("lstm", "cnn_lstm") for v in VARIANTS}
    expected |= {"persistence", "repeat_window"}
    missing = sorted(expected - set(table.run))
    status = (
        f"All eight {'pilot' if pilot else 'full-data'} neural runs and both baselines are complete."
        if not missing
        else ("INCOMPLETE: remaining runs: " + ", ".join(missing))
    )
    bootstrap_path = Path(cfg["paths"]["outputs"]) / "paired_patient_bootstrap.csv"
    comparisons = (
        pd.read_csv(bootstrap_path) if bootstrap_path.stat().st_size > 2 else pd.DataFrame()
    )
    best = table.sort_values("mae").iloc[0]
    findings = [
        f"Lowest observed record-weighted test MAE: **{best.run}**, {best.mae:.6f} mV.",
        "This is descriptive; test results were not used to tune or choose checkpoints.",
    ]
    neural = table.loc[~table.run.isin(["persistence", "repeat_window"])]
    if len(neural):
        winner = neural.sort_values("mae").iloc[0]
        findings.append(f"Lowest observed neural-model MAE: **{winner.run}**, {winner.mae:.6f} mV.")
        persistence = table.loc[table.run == "persistence"]
        if len(persistence):
            reference = float(persistence.iloc[0].mae)
            direction = "lower" if winner.mae < reference else "higher"
            findings.append(
                f"Its MAE is {direction} than persistence ({reference:.6f} mV). "
                "This is a descriptive comparison under the configured data/training budget, "
                "not evidence of a general clinical advantage."
            )
        if pilot:
            findings.append(
                "The pilot uses only five epochs and a small cohort. Do not infer "
                "architecture superiority or conditioning benefit from this run alone. "
                "Increase training patients and the predeclared epoch budget, preserving "
                "patient isolation and validation-based checkpoint selection."
            )
    for row in comparisons.itertuples(index=False):
        findings.append(
            f"- {row.effect}: {row.B} minus {row.A}: patient-macro MAE delta "
            f"{row.delta_patient_macro_mae_mV:+.6f} mV, 95% interval "
            f"[{row.ci95_low:+.6f}, {row.ci95_high:+.6f}]."
        )
    benchmark_path = root / "benchmark.json"
    timing = (
        markdown_table(pd.DataFrame(read_json(benchmark_path)))
        if benchmark_path.exists()
        else ("Benchmark not yet run.")
    )
    metric_table = table.drop(columns=["parameters"])
    methods = METHODS.replace(
        "Maximum 25 epochs, patience 5",
        f"Maximum {cfg['training']['epochs']} epochs, patience {cfg['training']['patience']}",
    )
    methods = methods.replace(
        "yield 17,418/2,183/2,198 records", "yield 17,418/2,183/2,198 records in the full dataset"
    )
    methods = methods.replace(
        "Available history covers 2,929 records",
        "In the full dataset, available history covers 2,929 records",
    )
    methods = methods.replace("only 293 test records have it", "293 full-test records have it")
    body = "# Contrastive Patient-Stratified ECG Forecasts\n\n" + status + "\n\n"
    cohort = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(Path(cfg["data"]["cache"]) / f"metadata_{split}.csv")
        cohort.append(
            {
                "split": split,
                "records": len(frame),
                "patients": frame.patient_id.nunique(),
                "with_prior_history": int(frame.history_available.sum()),
            }
        )
    if pilot:
        body += (
            "**Small-data pilot: exploratory results, not the full PTB-XL benchmark.** "
            "Whole patients were sampled uniformly within the official folds using seed "
            f"{cfg['subset']['seed']}. All available records of each selected patient were kept. "
            "Age and waveform scalers were refitted on selected training data only. "
            "Physical voltages were recovered from the full cache by inverse transformation "
            "before refitting (float32 rounding is possible). Small cohorts and short training "
            "budgets limit conclusions and literature comparisons.\n\n"
        )
    body += "## Active cohort\n\n" + markdown_table(pd.DataFrame(cohort)) + "\n\n" + methods
    body += "\n## Measured results\n\nErrors are in original mV (MSE in mV^2).\n\n"
    body += f"![Dataset audit]({asset('metadata_distributions.png')})\n\n"
    body += markdown_table(metric_table) + "\n\n## Observed comparisons\n\n" + "\n\n".join(findings)
    body += "\n\n## Local timing benchmark\n\n" + timing
    body += "\n\nBenchmark estimates exclude validation, checkpoint and plotting overhead; "
    body += "measured run times are in the results table.\n\n"
    body += LITERATURE
    body += "\n## Figures and provenance\n\n"
    for run in table.run:
        body += f"### {run}\n\n"
        body += f"![Representative forecast]({asset(run + '/forecast_lowest_id.png')})\n\n"
        body += f"![Per-lead errors]({asset(run + '/per_lead_errors.png')})\n\n"
        if (root / run / "history.csv").exists():
            body += f"![Learning curves]({asset(run + '/learning_curves.png')})\n\n"
    body += "Configs, feature manifests, histories, checkpoints, per-record errors and predictions "
    body += "are stored under experiments/<run>. Subgroup/horizon CSVs and remaining figures "
    config_name = config_path or ("configs/pilot.yaml" if pilot else "configs/base.yaml")
    body += f"are under the configured outputs directory. Re-run `python src/report.py --config {config_name}` to regenerate.\n"
    (docs / "ASSIGNMENT_REPORT.md").write_text(body, encoding="utf-8")
    notes = presentation_notes(table)
    notes = notes.replace("full-data runs", "runs for the configured cohort")
    notes = notes.replace("maximum 25", f"maximum {cfg['training']['epochs']}")
    notes = notes.replace("patience five", f"patience {cfg['training']['patience']}")
    if pilot:
        notes = notes.replace(
            "# Class presentation and speaker notes",
            "# Class presentation and speaker notes\n\n**Small-data pilot: preliminary results; see the active-cohort table in the assignment report.**",
        )
    (docs / "PRESENTATION_NOTES.md").write_text(notes, encoding="utf-8")
    print(status, flush=True)
    print(f"Report: {docs / 'ASSIGNMENT_REPORT.md'}", flush=True)
    return table


def bundle(cfg):
    """Explicit allowlist: never recursively zip the workspace or raw dataset."""
    destination = Path(cfg["paths"]["outputs"]) / "colab_bundle.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [Path(name) for name in ("pyproject.toml", "requirements.txt", "README.md")]
    for folder in (
        Path("src"),
        Path("configs"),
        Path("notebooks"),
        Path("tests"),
        Path(cfg["data"]["cache"]),
    ):
        files.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    allowed = {".py", ".toml", ".txt", ".md", ".yaml", ".ipynb", ".npy", ".json", ".csv"}
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(set(files)):
            if path.suffix.lower() not in allowed or not path.is_relative_to(
                Path.cwd() if path.is_absolute() else Path(".")
            ):
                continue
            if path.as_posix().endswith("src/report.py"):
                # The local manuscript's extracted results must not travel to Colab.
                source = path.read_text(encoding="utf-8")
                source = re.sub(
                    r'(?s)LITERATURE = """.*?"""',
                    'LITERATURE = "The private manuscript comparison is omitted from the Colab bundle. Regenerate the report locally for that section."',
                    source,
                )
                archive.writestr(path.as_posix(), source)
            else:
                archive.write(path, arcname=path.as_posix())
    print(
        f"Created {destination} ({destination.stat().st_size / 1e6:.1f} MB). "
        "Contains code/config/notebooks/preprocessed cache only; no PDF, raw data or checkpoints."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument(
        "--bundle", action="store_true", help="Build the local code/cache ZIP for Colab"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.bundle:
        bundle(config)
    else:
        report(config, config_path=args.config)
