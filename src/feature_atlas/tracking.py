"""Metric logging behind a small interface, so training never depends on one service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    def log(self, step: int, metrics: dict[str, float]) -> None: ...

    def close(self) -> None: ...


class JsonlTracker:
    """Appends one JSON object per log call.

    A resumed run appends again from its last checkpoint, so steps logged after that
    checkpoint appear twice. Readers should keep the last entry for each step.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a")

    def log(self, step: int, metrics: dict[str, float]) -> None:
        self._file.write(json.dumps({"step": step, **metrics}) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class NullTracker:
    """Discards everything; for tests."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        pass

    def close(self) -> None:
        pass


def read_metrics(path: Path) -> list[dict[str, float]]:
    """Read a JSONL log, keeping the last entry for each step."""
    latest: dict[int, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        latest[int(entry["step"])] = entry
    return [latest[step] for step in sorted(latest)]
