# Study guide: understanding the full experiment

A reading guide to the PTB-XL 12-lead ECG forecasting study (Phases 0–19). It shows
what to read, in which order, and what you should be able to explain afterwards. All
numbers below come from the phase reports; follow the links for the evidence.

## 1. Choose a reading path

| Time | Path |
|---|---|
| 15 min | §2 (the story) and §6 (key numbers) of this guide |
| 1 hour | [README](../README.md) phase table → [FINAL_REPORT.md](FINAL_REPORT.md) (abstract, §5 Discussion, §6 Limitations, §8) → the [presentation](presentation/PTBXL_ECG_forecasting.pptx) with speaker notes open |
| 1 day | All of the above, then every phase in the order of §4, reading `PREREGISTRATION.md` before `REPORT.md`, then the [decision log](DECISION_LOG.md) |

The speaker notes of every slide follow the same structure (purpose, key
interpretation, discussion points, details not on the slide, likely questions), so the
deck doubles as a compact study text.

## 2. The story in six steps

If you can explain each step in one sentence, you understand the experiment.

1. **Shape repeats, timing drifts.** In 7 s of ECG, beat shape is very predictable, but
   beat timing beyond about 1 s is not. Perfect timing would cut MAE by ≈ 34%
   ([Phase 0](phase0_feasibility/REPORT.md)).
2. **Pointwise error rewards blur.** Under MAE, a flat line (0.110 mV) beats repeating
   real beats (0.118 mV), because a well-shaped beat that is slightly late is punished
   twice ([Phase 0](phase0_feasibility/REPORT.md)).
3. **So MAE alone is the wrong judge.** Every pointwise metric ranks a perfect ECG that
   is 50 ms late below a flat line. The study uses MAE plus three *gatekeepers*
   (DTW-MAE, template MAE, beat timing) ([Phase 6](phase6_evaluation/REPORT.md)).
4. **Data, preprocessing and architecture do not fix it.** Filters, scalers,
   augmentations, more data and five model families all leave the beats blurred: with an
   L1 loss, no model forecasts a QRS complex ([Phases 5](phase5_preprocessing/REPORT.md),
   [7](phase7_data_efficiency/REPORT.md), [9](phase9_models/REPORT.md)).
5. **The objective does.** Giving a CNN-LSTM the causal `sn4` forecast (repeat the last
   four beats) through a learned gate is the first change that improves MAE *and* every
   gatekeeper ([Phase 10](phase10_objectives/REPORT.md)).
6. **It holds, with known limits.** It holds across folds and subgroups and on the
   sealed test fold, but it does not beat `sn4` on beat shape or timing, fails for
   AF/flutter, and damps beats to ≈ 28% amplitude
   ([Phases 12–18](FINAL_REPORT.md#8-phases-1019-including-the-fold-10-test-2026-11-24)).

## 3. Glossary

| Term | Meaning |
|---|---|
| Task | Observe the first 7 s (700 samples × 12 leads, 100 Hz) and forecast the next 2.9 s (290 samples), in raw mV |
| 2.9 s, not 3 s | The last 10 samples of every official 100 Hz file are a 500→100 Hz conversion artefact (up to 0.15 mV), so they are trimmed (D3.4) |
| Flat line | Forecast = median of the history; the baseline to beat on MAE |
| `sn4` | Seasonal-naive baseline: estimate the cycle length from the history, average the last 4 cycles (level-matched), repeat forward. Causal, no learning. The baseline to beat on the gatekeepers |
| MAE | Primary metric: mean absolute error, 12 leads, averaged per patient (patient-macro) |
| DTW-MAE | Gatekeeper 1: shape error allowing a ±100 ms time shift |
| Template MAE | Gatekeeper 2: error of the median beat (morphology) |
| Beat timing | Gatekeeper 3: capped symmetric distance between forecast and true R-peaks (ms); also reported as the share of records ≤ 50 ms |
| D6.2 rule | A beats B only if the 95% CI of ΔMAE is entirely below 0 **and** no gatekeeper's CI is entirely on the worse side |
| `compare_runs` | Comparison with 6 training runs per arm and a bootstrap over runs and patients (D8.3) |
| Fold 9 | Model-selection fold (official PTB-XL fold) |
| 9-fold CV | Confirmation over folds 1–9; OOF = out-of-fold forecasts |
| Fold 10 | Sealed test fold, opened once in Phase 17 |
| Internal holdout | 20% of training patients, used for early stopping and quick screening |
| sn4-gated CNN-LSTM | Final model: forecast = σ(gate(h)) ⊙ `sn4` + correction(h), where h is the CNN-LSTM's summary of the history ([code](../src/ptbxl_forecast/models/gated_baseline.py)) |
| Ablation | A copy of the final model with one component removed, to measure its contribution |

## 4. Phase by phase

Read each phase as *question → method → result → decision*. The decision IDs refer to
the [decision log](DECISION_LOG.md).

| Phase | Question | Key result | Decision |
|---|---|---|---|
| [0 Feasibility](phase0_feasibility/REPORT.md) | How predictable is 7 s → 3 s? | Shape predictable, timing not; flat beats `sn4` on MAE | Keep the task, change the evaluation (D0.8) |
| [1 Foundation](phase1_foundation/REPORT.md) | Can every run be reproduced? | uv lockfile, validated config, per-run provenance, bitwise reruns | D1.1–D1.8 |
| [1.5 Data](phase1_5_data/REPORT.md) | Is the download complete and authentic? | All files match SHA-256; upstream `RECORDS` file is malformed | Never use `RECORDS` (D1.5.3) |
| [2 Dataset](phase2_dataset/REPORT.md) | What is in the data? | 8 independent leads; device drives difficulty; fold 9 device-shifted | D2.3–D2.5 |
| [3 Quality](phase3_quality/REPORT.md) | Which records are trustworthy? | 30 defect records; 100 ms end artefact; baseline wander drives difficulty | Exclude and trim → 2.9 s (D3.1, D3.4) |
| [4 Leakage](phase4_leakage/REPORT.md) | Which splits and inputs are safe? | Record splits leak patients; labels and zero-phase filters leak the future | Patient-level folds, causal inputs, fold 10 sealed in code (D4.1–D4.4) |
| [6 Evaluation](phase6_evaluation/REPORT.md) | How should forecasts be scored? | All pointwise metrics fail the 50 ms test | MAE + 3 gatekeepers (D6.2) |
| [5 Preprocessing](phase5_preprocessing/REPORT.md) | Do filters, scalers, features or augmentation help? | All effects < 1 µV; reference CNN beats flat but not `sn4` | Raw mV + 0.67 Hz baseline/residual input (D5.11) |
| [7 Data size](phase7_data_efficiency/REPORT.md) | Does more data help? | MAE gains diminish (floor ≈ 0.094 mV); timing ≈ 258 ms at every size | Use the full pool; seeding bug fixed (D7.1) |
| [8 Validation](phase8_validation/REPORT.md) | How do we avoid false wins? | Old rule: 41% false positives; 6 runs/arm + run-and-patient bootstrap: 2.8% | `compare_runs` (D8.3); select on fold 9, confirm with CV (D8.6) |
| [9 Models](phase9_models/REPORT.md) | Which architecture is best under L1? | CNN-LSTM best MAE (−7.4 µV vs CNN, 9/9 folds) but worse morphology; none beats `sn4` | No winner (D9.3); CNN-LSTM + Transformer go on (D9.4) |
| [10 Objectives](phase10_objectives/REPORT.md) | Can the loss fix shape and timing? | `sn4`-gated CNN-LSTM beats flat and L1 on all 4 metrics (9/9 folds) | Adopted (D10.2); Transformer dropped for compute (D10.0) |
| [11 Tuning](phase11_tuning/REPORT.md) | Which hyperparameters? | Hidden 256: −0.77 µV MAE on fold 9 | Adopted, later reverted (D11.1 → D12.1) |
| [12 Robustness](phase12_robustness/REPORT.md) | Does it hold everywhere? | Beats flat in every stratum; ties L1 in AF/flutter; fails with lead II missing; CV shows hidden 256 hurts shape/timing | Revert to hidden 128 (D12.1) |
| [13 Plausibility](phase13_plausibility/REPORT.md) | Are the beats realistic? | Correct beat count; damped to ≈ 28% amplitude; limb-lead relations broken in 19% of records | Finding |
| [14 Patient errors](phase14_patient_errors/REPORT.md) | Which patients are hard? | AF/flutter, hypertrophy, pacing, baseline wander; device effect vanishes after adjustment | Finding |
| [15 Signal errors](phase15_signal_errors/REPORT.md) | Which signals are hard? | Gain over L1 only for periodic histories; gate tracks periodicity (ρ 0.47) | Finding |
| [16 Final model](phase16_final/REPORT.md) | What exactly is tested? | Hidden 128, 6 seeds, weights hashed in `MANIFEST.json` | Frozen before fold 10 |
| [17 Test](phase17_test/REPORT.md) | Does it generalise? | Fold 10 MAE 0.0850 mV; beats flat and L1 on every metric; worse than `sn4` on template and timing | Reported once (D17.1) |
| [18 Ablations](phase18_ablations/REPORT.md) | Which parts matter? | Skip path essential; gate trades shape for MAE; raw input better on fold 9 | Frozen model unchanged (D18.1) |
| 19 Documentation | – | [FINAL_REPORT.md](FINAL_REPORT.md) | – |

Phase 6 ran before Phase 5 (D5.0): preprocessing could only be compared once the
metrics were fixed.

## 5. The final model in one paragraph

A 7 s history (24 channels: causal 0.67 Hz baseline + residual for each lead) passes
through two strided 1-D convolutions (700 → 175 steps) and a 2-layer LSTM (hidden 128).
Its last hidden state h feeds two linear heads: a sigmoid **gate** (one value per
horizon step and lead) and a **correction**. The forecast is gate × `sn4` + correction.
A closed gate gives the plain L1 CNN-LSTM; an open gate with no correction gives `sn4`.
The network never has to learn beat timing, because `sn4` supplies it; it only learns how
far to trust it. 1.17 M parameters, L1 loss, AdamW (lr 1e-3, batch 64), early stopping
on the internal holdout; the final forecast is the mean of 6 seeds.

## 6. Key numbers to remember

| Quantity | Value | Source |
|---|---|---|
| Clean development set | 19,571 records, 16,945 patients (folds 1–9) | Phase 3 |
| Flat / `sn4` MAE (9-fold CV) | 0.1097 / 0.1167 mV | Phase 9 |
| `sn4` template / timing | 0.065 mV / 88 ms | Phase 9 |
| L1 CNN-LSTM (CV) | MAE 0.0866, template 0.131, timing 270 ms | Phase 9 |
| Final model (CV) | MAE 0.0850, template 0.088, timing 109 ms, 36% of records ≤ 50 ms | Phase 10 |
| Final model (fold 10) | MAE 0.0850 mV; vs L1 −1.7 µV [−2.6, −0.9], template −41 µV, timing −145 ms | Phase 17 |
| Final vs `sn4` (fold 10) | template +27 µV, timing +32 ms (worse) | Phase 17 |
| Without the gate | +16 µV MAE, but `sn4`-quality shape and timing | Phase 18 |
| `sn4` as input only | timing 264 ms (no better than without `sn4`) | Phase 18 |

1 µV = 0.001 mV. Model differences are often a few µV against an MAE of ≈ 85 µV, which is
why the statistics (Phase 8) matter.

## 7. Presentation slide map

| Slides | Topic | Main report |
|---|---|---|
| 1–3 | Title, problem, research questions | Phase 0, [FINAL_REPORT.md](FINAL_REPORT.md) §1 |
| 4 | Feasibility | Phase 0 |
| 5–7 | Dataset, quality, leakage | Phases 2, 1.5, 3, 4, 8 |
| 8–10 | Preprocessing, evaluation, data size | Phases 5, 6, 7 |
| 11–15 | Model families (roadmap, LSTM, CNN-LSTM, PatchTST, Transformer) | Phase 9 |
| 16–18 | Objectives, final architecture, tuning | Phases 10, 11, 16 |
| 19 | Statistical reliability | Phase 8 |
| 20–25 | Robustness, error analysis, leads, plausibility, explainability, ablations | Phases 12–15, 18 |
| 26–28 | Compute, fold-10 test, final comparison | Phases 9, 17 |
| 29–32 | Decisions, conclusions, limitations, future work | [DECISION_LOG.md](DECISION_LOG.md), [FINAL_REPORT.md](FINAL_REPORT.md) §6 |

## 8. Read the code

| What | Where |
|---|---|
| Final model | [`src/ptbxl_forecast/models/gated_baseline.py`](../src/ptbxl_forecast/models/gated_baseline.py) |
| CNN-LSTM body | [`src/ptbxl_forecast/models/cnn_lstm.py`](../src/ptbxl_forecast/models/cnn_lstm.py) |
| `sn4` (period estimate, repeated cycles) | `estimate_period`, `seasonal` in [`research/phase0_feasibility.py`](../research/phase0_feasibility.py) |
| Data access, sealed fold 10 | `src/ptbxl_forecast/data.py` |
| Leakage guards and splits | `src/ptbxl_forecast/leakage.py`, `src/ptbxl_forecast/splits.py` |
| Each phase's analysis | `research/phase<N>_*.py` (the command is at the top of each report) |

## 9. Test yourself

Try to answer before opening the answer.

<details><summary>Why is the horizon 2.9 s and not 3 s?</summary>

The last 10 samples of every official 100 Hz file are a 500→100 Hz conversion artefact
(up to 0.15 mV, checked against the 500 Hz originals). They were trimmed for every fold
(D3.4).
</details>

<details><summary>Why can't MAE alone choose the best model?</summary>

Beat timing is uncertain, so the MAE-optimal forecast averages over possible beat
positions, which erases beats. A flat line beats a correctly shaped beat that is 50 ms
late. The gatekeepers stop a blurred model from winning (Phase 6, D6.2).
</details>

<details><summary>Why 6 runs per arm?</summary>

Training randomness changes verdicts at the sub-µV level. The old protocol declared a
false "win" in 41% of comparisons between identical configurations; 6 runs per arm with
a run-and-patient bootstrap brought this to 2.8% (Phase 8, D8.3).
</details>

<details><summary>Why was hidden 256 switched back to 128?</summary>

Hidden 256 won on fold 9 (MAE −0.77 µV), but 9-fold CV showed worse template MAE
(+1.3 µV, 8/9 folds) and timing (+5.2 ms, 9/9 folds). The revert used folds 1–9 only,
before fold 10 was opened (D12.1).
</details>

<details><summary>Why does the gate damp the beats?</summary>

Under L1, when timing is uncertain, a damped beat has a lower expected error than a
full beat at a possibly wrong time. Without the gate the model keeps full beats
(`sn4`-quality shape and timing) but loses 16 µV MAE (Phase 18).
</details>

<details><summary>Is using sn4 "cheating"?</summary>

No. `sn4` is computed from the 7 s history only, which every model sees. It is a
hand-crafted causal feature with a skip connection to the output.
</details>

<details><summary>Where does the final model fail?</summary>

AF/flutter (no gain over L1, timing worse than `sn4`), a missing lead II (template worse
than flat), limb-lead consistency (broken in 19% of records because the gate is per
lead), and amplitude (beats damped to ≈ 28%). It never beats `sn4` on template MAE or
timing.
</details>

<details><summary>What contributed most to the result?</summary>

The `sn4` skip path (removing it: +2.9 µV MAE, +35 µV template, +140 ms timing), then
architecture (CNN-LSTM vs CNN: −7.4 µV MAE), far ahead of preprocessing and data size
(< 1 µV).
</details>

<details><summary>What would you do next?</summary>

A periodicity-aware gate with a fallback for irregular rhythms; lead-consistent output;
probabilistic forecasts; timing-tolerant objectives and full-size attention models on a
GPU; external validation on another dataset (fold 10 is now used).
</details>

## 10. Reproduce

The setup and per-phase commands are in the [README](../README.md). Each phase report
starts with the exact command that produced it, and [FINAL_REPORT.md §9](FINAL_REPORT.md#9-reproducing-this-report)
lists the full sequence.
