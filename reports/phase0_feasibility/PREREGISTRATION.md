# Phase 0 pre-registration: forecasting feasibility

Written 2026-11-22, before any Phase 0 analysis was run. Results are reported in
[REPORT.md](REPORT.md) and must be judged against these criteria, not criteria
chosen afterwards.

## Scope

- PTB-XL v1.0.3, native 100 Hz, 12 leads, raw mV, `strat_fold` 1–9 only.
  Fold 10 stays sealed for the final test and is never loaded in Phase 0.
- No learned models. Phase 0 measures how much structure the signal itself
  offers, using only non-learned, per-record forecasts and oracle diagnostics.
- Primary error: MAE in mV over 12 leads × 300 future samples, averaged per
  patient first (patient-macro) so repeat patients do not dominate. Intervals:
  95% patient-cluster bootstrap, 1,000 resamples.
- Skill relative to the per-record history-mean forecast:
  `skill = 1 − MAE(method) / MAE(history mean)`.

## Forecasts compared

| ID | Forecast | Uses future? |
|---|---|---|
| `zero` | 0 mV | no |
| `mean` | per-lead mean of the 7 s history ("climatology") | no |
| `last` | last observed value (persistence) | no |
| `lastsec` | per-lead mean of the final observed second | no |
| `sn1` | seasonal naive: repeat the last cardiac cycle, period from history | no |
| `sn4` | mean of up to 4 previous cycles, level-matched to the last cycle | no |
| `oracle` | history beat morphology placed at the **true** future R-peaks | **yes (diagnostic only)** |

## Hypotheses and acceptance criteria

**H0.1 — Exploitable structure exists.** The best non-learned structural forecast
(`sn4`) beats the history mean over the 3 s horizon.
Accept if the paired patient-bootstrap 95% CI of MAE(`sn4`) − MAE(`mean`) is
entirely below 0.

**H0.2 — Skill persists across the full 3 s horizon.** Accept if the skill of
`sn4` over the final second of the horizon (lead time 2–3 s) is > 0 with its
95% CI excluding 0. Otherwise, a 3 s horizon is beyond the structural
predictability horizon of the signal for pointwise metrics.

**H0.3 — Timing, not morphology, dominates forecast error.** Accept if knowing
the true future beat times (`oracle`) reduces MAE by ≥ 25% relative to `sn4`.

**H0.4 — Predictability depends on rhythm.** Accept if (a) whole-horizon skill
for AFIB is lower than for SR with non-overlapping 95% CIs, and (b) Spearman
ρ between per-record skill and history RR coefficient of variation is ≤ −0.3.

**H0.5 — Lead time, not context length, governs difficulty.**
(a) At matched lead time (first 1 s of each horizon), `sn4` MAE differs by
< 5% between the 7→3, 8→2 and 9→1 splits.
(b) Predicting the fixed window 9–10 s from 7 s of context improves `sn4` MAE
by < 5% compared with 3 s of context.

**H0.6 — Earlier neural models learned little beyond climatology.** Accept if
the previously reported Week 5 validation pooled MAE (0.1185 mV, fold 9, 100
patients) is within 5% of the `mean` forecast's pooled MAE on fold 9 and not
better than `sn4`. Caveat: this compares against reported numbers on a
subset; it is indicative, not a controlled rerun.

## Go / no-go rule for Phase 1

- **Go, as specified (7→3):** H0.1 and H0.2 accepted.
- **Go with revised task/metrics:** H0.1 accepted, H0.2 rejected. The project
  must then adopt evaluation that separates rhythm (timing) from morphology,
  and/or a horizon within the measured skill horizon.
- **No-go:** H0.1 rejected (no structure beyond the mean is exploitable).
