"""Explicit finite experiment stages, reusing the same training/evaluation functions."""

import copy

from config import device_for, file_hash, read_json, study_dir, write_json
from data import prepare
from evaluate import (
    baseline,
    evaluate_run,
    final_evaluation,
    freeze_selection,
    outputs,
    ranking_key,
    verify_evaluation,
)
from feature_selection import mrmr, permutation
from leakage_audit import audit
from train import train_run
from visualize import diagnostic_figures, run_figures


def condition(
    cfg,
    cache,
    model,
    filtering,
    features,
    *,
    resume=False,
    lr=None,
    hidden=None,
    selection=None,
    name=None,
):
    budget = (
        cfg["training"]["pilot_epochs"]
        if cfg["run_mode"] == "pilot"
        else (
            cfg["training"]["full_epochs"]
            if cfg["run_mode"] == "full"
            else cfg["training"]["epochs"]
        )
    )
    rate, size = lr or cfg["training"]["lr"], hidden or cfg["model"]["hidden"]
    name = name or f"{model}_{filtering}_{features}_lr{rate:g}_h{size}_e{budget}"
    run = study_dir(cfg) / cfg["run_mode"] / "checkpoints" / name
    complete = (run / "status.json").exists() and read_json(run / "status.json")[
        "state"
    ] == "completed"
    # Completed runs may be reused, but train_run first checks the full scientific signature.
    run = train_run(
        cfg,
        cache,
        model,
        filtering,
        features,
        name=name,
        lr=rate,
        hidden=size,
        resume=complete or (resume and (run / "last.pt").exists()),
        selection=selection,
    )
    receipt = outputs(cfg, run.name) / "val.json"
    if not complete or not receipt.exists():
        evaluate_run(cfg, cache, run)
    else:
        verify_evaluation(receipt.parent, "val")
        result = read_json(receipt)
        if result["checkpoint_sha256"] != file_hash(run / "best.pt"):
            raise ValueError("Completed run's validation evidence does not match its checkpoint")
    return run, read_json(receipt)


def smoke(cfg, resume=False):
    cfg = copy.deepcopy(cfg)
    cfg["run_mode"] = "pilot"
    cache, manifest = prepare(cfg)
    audit(cfg, cache)
    runs = []
    for model in ("lstm", "cnn_lstm"):
        for filtering in ("P0", "P1"):
            run, _ = condition(cfg, cache, model, filtering, "F0", resume=resume)
            runs.append(run)
        run, _ = condition(cfg, cache, model, "P0", "F1", resume=resume)
        runs.append(run)
    for name in ("persistence", "repeat_window"):
        baseline(cfg, cache, name)
    diagnostic_figures(cfg, cache)
    for run in (runs[0], runs[3]):
        run_figures(cfg, cache, run)
    result = {
        "passed": True,
        "cohort": manifest["cohort"],
        "splits": manifest["splits"],
        "runs": [r.name for r in runs],
        "epochs_each": cfg["training"]["pilot_epochs"],
        "test_waveforms_opened": False,
        "research_performance_claim": False,
        "synthetic": manifest["synthetic"],
    }
    write_json(study_dir(cfg) / "pilot" / "smoke_checks.json", result)
    return result


def screen(cfg, resume=False):
    cfg = copy.deepcopy(cfg)
    cfg["run_mode"] = "medium"
    cache, manifest = prepare(cfg)
    audit(cfg, cache)
    selected = {}
    for model in ("lstm", "cnn_lstm"):
        candidates = []
        for filtering in ("P0", "P1"):
            _, result = condition(cfg, cache, model, filtering, "F0", resume=resume)
            candidates.append(result)
        best_filter = min(candidates, key=ranking_key)
        _, conditioned = condition(
            cfg, cache, model, best_filter["preprocessing"], "F1", resume=resume
        )
        best = min([best_filter, conditioned], key=ranking_key)
        selected[model] = {
            "preprocessing": best["preprocessing"],
            "feature_set": best["feature_set"],
            "run": best["name"],
            "lr": cfg["training"]["lr"],
            "hidden": cfg["model"]["hidden"],
        }
    result = {"cohort": manifest["cohort"], "selected": selected, "used_test": False}
    write_json(study_dir(cfg) / "medium" / "screening_selection.json", result)
    for name in ("persistence", "repeat_window"):
        baseline(cfg, cache, name)
    diagnostic_figures(cfg, cache)
    return result


def tune(cfg, resume=False):
    cfg = copy.deepcopy(cfg)
    cfg["run_mode"] = "medium"
    cache, manifest = prepare(cfg)
    previous = read_json(study_dir(cfg) / "medium" / "screening_selection.json")
    if previous["cohort"] != manifest["cohort"]:
        raise ValueError("Screening used a different cohort")
    selected = {}
    for model, recipe in previous["selected"].items():
        candidates = []
        for rate in cfg["training"]["learning_rates"]:
            run, result = condition(
                cfg,
                cache,
                model,
                recipe["preprocessing"],
                recipe["feature_set"],
                resume=resume,
                lr=rate,
            )
            candidates.append((run, result))
        if cfg["training"]["tune_hidden_128"]:
            best_run, _ = min(candidates, key=lambda x: ranking_key(x[1]))
            rate = read_json(best_run / "run.json")["config"]["training"]["lr"]
            candidates.append(
                condition(
                    cfg,
                    cache,
                    model,
                    recipe["preprocessing"],
                    recipe["feature_set"],
                    resume=resume,
                    lr=rate,
                    hidden=128,
                )
            )
        run, result = min(candidates, key=lambda x: ranking_key(x[1]))
        record = read_json(run / "run.json")
        selected[model] = {
            "preprocessing": recipe["preprocessing"],
            "feature_set": recipe["feature_set"],
            "lr": record["config"]["training"]["lr"],
            "hidden": record["config"]["model"]["hidden"],
            "validation_run": result["name"],
        }
    result = {"cohort": manifest["cohort"], "selected": selected, "used_test": False}
    write_json(study_dir(cfg) / "medium" / "tuning_selection.json", result)
    return result


def selected_recipes(cfg):
    path = study_dir(cfg) / "medium" / "tuning_selection.json"
    if not path.exists():
        raise ValueError("Complete medium screening and tuning before extensions/full training")
    result = read_json(path)
    if result.get("used_test") is not False:
        raise ValueError("Selection must use validation only")
    medium = copy.deepcopy(cfg)
    medium["run_mode"] = "medium"
    _, current = prepare(medium)
    if result["cohort"] != current["cohort"]:
        raise ValueError("Selected recipes used a different source/cohort/protocol")
    for recipe in result["selected"].values():
        run = study_dir(cfg) / "medium" / "checkpoints" / recipe["validation_run"]
        saved = read_json(run / "run.json")
        actual = (
            saved["preprocessing"],
            saved["feature_set"],
            saved["config"]["training"]["lr"],
            saved["config"]["model"]["hidden"],
        )
        if actual != (
            recipe["preprocessing"],
            recipe["feature_set"],
            recipe["lr"],
            recipe["hidden"],
        ):
            raise ValueError("Selected recipe no longer matches its validation run")
    return result["selected"]


def feature_extension(cfg, resume=False):
    cfg = copy.deepcopy(cfg)
    cfg["run_mode"] = "medium"
    cache, manifest = prepare(cfg)
    saved = read_json(study_dir(cfg) / "medium" / "tuning_selection.json")
    if saved["cohort"] != manifest["cohort"]:
        raise ValueError("Tuning used a different cohort")
    ranking = mrmr(cfg, cache)
    completed = []
    for model, recipe in saved["selected"].items():
        expanded, _ = condition(
            cfg,
            cache,
            model,
            recipe["preprocessing"],
            "expanded",
            resume=resume,
            lr=recipe["lr"],
            hidden=recipe["hidden"],
        )
        permuted = permutation(cfg, cache, expanded)
        for method, selection in (("mrmr", ranking), ("permutation", permuted)):
            run, _ = condition(
                cfg,
                cache,
                model,
                recipe["preprocessing"],
                method,
                resume=resume,
                lr=recipe["lr"],
                hidden=recipe["hidden"],
                selection=selection,
            )
            completed.append(run.name)
    result = {"completed": completed, "cohort": manifest["cohort"], "used_test": False}
    write_json(study_dir(cfg) / "medium" / "feature_extension.json", result)
    return result


def full(cfg, resume=False):
    cfg = copy.deepcopy(cfg)
    cfg["run_mode"] = "full"
    device_for(cfg)
    recipes = selected_recipes(cfg)
    cache, _ = prepare(cfg)
    audit(cfg, cache)
    for name in ("persistence", "repeat_window"):
        baseline(cfg, cache, name)
    diagnostic_figures(cfg, cache)
    runs = []
    for model, recipe in recipes.items():
        for filtering in ("P0", "P1"):
            run, _ = condition(
                cfg,
                cache,
                model,
                filtering,
                recipe["feature_set"],
                resume=resume,
                lr=recipe["lr"],
                hidden=recipe["hidden"],
            )
            runs.append(run)
            run_figures(cfg, cache, run)
            # Preserve validation evidence at the exact requested epoch milestones.
            for epoch in (60, 120):
                if (run / f"best_through_{epoch}.pt").exists():
                    evaluate_run(
                        cfg,
                        cache,
                        run,
                        checkpoint=f"best_through_{epoch}.pt",
                        label=run.name + f"_through_{epoch}",
                    )
    freeze_selection(cfg, cache, runs)
    if cfg["evaluation"]["final_test"]:
        final_evaluation(cfg, cache)
        for run in runs:
            run_figures(cfg, cache, run, "test")
    return {"runs": [run.name for run in runs], "test_requested": cfg["evaluation"]["final_test"]}
