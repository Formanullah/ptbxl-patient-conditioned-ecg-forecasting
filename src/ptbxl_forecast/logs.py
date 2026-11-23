"""Standard-library logging: console for humans, a per-run file for the record."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the root logger. Safe to call repeatedly (handlers are replaced)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level.upper())
    console.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
        root.addHandler(file_handler)

    logging.captureWarnings(True)
    for noisy in ("matplotlib", "PIL", "urllib3", "alembic", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("ptbxl_forecast")
