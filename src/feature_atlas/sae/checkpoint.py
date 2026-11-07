"""Atomic training checkpoints.

Free GPU sessions die without warning, so a checkpoint is written to a temporary file
and renamed into place. A kill mid-write leaves the previous checkpoint intact.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_NAME = re.compile(r"step_(\d{9})\.pt")


class CheckpointMismatchError(ValueError):
    """A checkpoint was written under a different training config."""


class CheckpointManager:
    """Saves, finds and prunes checkpoints for one run."""

    def __init__(self, directory: Path, training_hash: str, keep: int = 2) -> None:
        self.directory = directory
        self.training_hash = training_hash
        self.keep = keep

    def all(self) -> list[Path]:
        """Checkpoints in step order, oldest first."""
        if not self.directory.exists():
            return []
        found = [path for path in self.directory.iterdir() if _NAME.fullmatch(path.name)]
        return sorted(found, key=lambda path: path.name)

    def latest(self) -> Path | None:
        found = self.all()
        return found[-1] if found else None

    def save(self, step: int, payload: dict[str, Any]) -> Path:
        """Write ``payload`` for ``step`` atomically, then delete all but the newest few."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"step_{step:09d}.pt"
        temporary = path.with_name(path.name + ".tmp")
        torch.save({**payload, "step": step, "training_hash": self.training_hash}, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        for old in self.all()[: -self.keep]:
            old.unlink()
        logger.info("saved checkpoint", extra={"fields": {"step": step, "path": str(path)}})
        return path

    def load(self, path: Path) -> dict[str, Any]:
        """Load a checkpoint to CPU, refusing one from a different config."""
        # weights_only=True refuses arbitrary pickled objects, so a tampered file can't run code.
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("training_hash") != self.training_hash:
            raise CheckpointMismatchError(
                f"{path} was written with training hash {payload.get('training_hash')}, but "
                f"this config has {self.training_hash}. Use a new run name to start fresh."
            )
        return payload
