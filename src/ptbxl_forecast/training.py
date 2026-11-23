"""Generic training loop for direct forecasters (early stopping on an internal holdout).

A *pipeline* converts raw records into model inputs/targets and model outputs back to
raw 12-lead mV:

    pipeline.inputs(x_raw)                -> model input (n, T, C)       (history only)
    pipeline.targets(y_raw, x_raw)        -> model target (n, F, L')
    pipeline.outputs(pred, x_raw)         -> raw 12-lead forecast (n, F, 12)
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .seeding import seed_everything, torch_generator


def train_forecaster(
    model: torch.nn.Module,
    pipeline,
    rec_train: np.ndarray,
    rec_holdout: np.ndarray,
    history: int,
    horizon: int,
    seed: int,
    augment=None,
    epochs: int = 40,
    patience: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
    threads: int = 4,
    reinit: bool = True,
) -> dict:
    """Train with L1 loss in the pipeline's target space. Returns the training record and
    leaves ``model`` holding the weights of the best holdout epoch.

    With ``reinit`` (default) every layer is re-initialised after seeding, so the initial
    weights depend only on ``seed`` and not on the RNG state when the model was built.
    Pass ``reinit=False`` to fine-tune existing weights."""
    seed_everything(seed)
    if reinit:
        for m in model.modules():
            # nn.MultiheadAttention only exposes the private _reset_parameters.
            init = getattr(m, "reset_parameters", None) or getattr(m, "_reset_parameters", None)
            if callable(init):
                init()
    torch.set_num_threads(threads)
    g = torch_generator(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xh, yh = rec_holdout[:, :history], rec_holdout[:, history : history + horizon]
    hold_in = torch.from_numpy(pipeline.inputs(xh).astype(np.float32))
    hold_out = torch.from_numpy(pipeline.targets(yh, xh).astype(np.float32))
    train = torch.from_numpy(rec_train[:, : history + horizon].astype(np.float32))
    best, best_state, bad, log = np.inf, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train), generator=g)
        losses = []
        for s in range(0, len(train), batch_size):
            rec = train[perm[s : s + batch_size]].clone()
            if augment is not None:
                rec = augment(rec, history, g)
            x, y = rec[:, :history].numpy(), rec[:, history:].numpy()
            inp = torch.from_numpy(pipeline.inputs(x).astype(np.float32))
            tgt = torch.from_numpy(pipeline.targets(y, x).astype(np.float32))
            opt.zero_grad()
            loss = torch.nn.functional.l1_loss(model(inp), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val = float(np.mean([torch.nn.functional.l1_loss(model(hold_in[s : s + 512]), hold_out[s : s + 512]).item()
                                 for s in range(0, len(hold_in), 512)]))  # fmt: skip
        log.append({"epoch": epoch, "train_l1": float(np.mean(losses)), "holdout_l1": val})
        if val < best - 1e-6:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"log": log, "best_holdout_l1": best, "best_epoch": int(np.argmin([r["holdout_l1"] for r in log]) + 1),
            "epochs_run": len(log), "seconds": time.perf_counter() - t0}  # fmt: skip


def predict(
    model: torch.nn.Module, pipeline, x_raw: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x_raw), batch_size):
            xb = x_raw[s : s + batch_size]
            pred = model(torch.from_numpy(pipeline.inputs(xb).astype(np.float32))).numpy()
            out.append(pipeline.outputs(pred.astype(np.float64), xb))
    return np.concatenate(out)
