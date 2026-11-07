"""Environment helpers: seeding, device selection, logging and paths."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, including any GPUs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(preferred: str = "auto") -> torch.device:
    """Return ``preferred``, or the fastest available device for ``"auto"``."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def artifacts_root() -> Path:
    """Return the directory for large local artifacts.

    ``FEATURE_ATLAS_HOME`` wins. Otherwise Kaggle and Colab use their writable working
    disks, and a laptop uses the current directory, whose artifact folders are ignored
    by git.
    """
    configured = os.environ.get("FEATURE_ATLAS_HOME")
    if configured:
        return Path(configured)
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Path("/kaggle/working")
    if os.environ.get("COLAB_RELEASE_TAG"):
        return Path("/content")
    return Path.cwd()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


def configure_logging(level: int = logging.INFO) -> None:
    """Send one JSON object per line to stderr, so runs can be parsed afterwards.

    Only this package logs at ``level``. Libraries log warnings and above, because HTTP
    clients log every request, which would bury training progress.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    logging.getLogger("feature_atlas").setLevel(level)
