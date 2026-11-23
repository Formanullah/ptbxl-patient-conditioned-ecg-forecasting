"""Seeding and deterministic execution for Python, NumPy and (optionally) PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> np.random.Generator:
    """Seed every RNG in use and return a NumPy Generator for explicit use.

    With ``deterministic=True`` PyTorch is restricted to deterministic kernels and
    raises if an operation has none, rather than silently giving run-to-run
    differences. ``PYTHONHASHSEED`` only affects interpreters started afterwards
    (e.g. spawned DataLoader workers).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return np.random.default_rng(seed)

    # Must be set before the first cuBLAS call for deterministic GEMMs on CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)  # also seeds all CUDA devices
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    return np.random.default_rng(seed)


def torch_generator(seed: int):
    """Generator for DataLoader shuffling, independent of the global RNG stream.

    No worker_init_fn is needed: PyTorch >= 1.9 seeds `random`, NumPy and torch in
    each worker from this generator (verified by tests/test_seeding.py).
    """
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def configure_torch(threads: int = 0, device: str = "auto") -> str:
    """Apply the thread count and resolve the device.

    CPU results are bitwise reproducible only at a fixed thread count (measured: 1, 4,
    8 and 16 threads give four different weight vectors), so set it explicitly.
    """
    import torch

    if threads > 0:
        torch.set_num_threads(threads)
    return resolve_device(device)


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
