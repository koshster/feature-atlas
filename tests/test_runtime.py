import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_atlas.runtime import artifacts_root, configure_logging, seed_everything, select_device


def test_seed_everything_repeats_all_generators() -> None:
    seed_everything(5)
    first = (np.random.rand(), torch.rand(1).item())
    seed_everything(5)
    assert first == (np.random.rand(), torch.rand(1).item())


def test_explicit_device_is_respected() -> None:
    assert select_device("cpu") == torch.device("cpu")


def test_env_var_overrides_artifact_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FEATURE_ATLAS_HOME", str(tmp_path))
    assert artifacts_root() == tmp_path


def test_kaggle_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FEATURE_ATLAS_HOME", raising=False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert artifacts_root() == Path("/kaggle/working")


def test_logs_are_json_with_extra_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    logging.getLogger("feature_atlas.test").info("done", extra={"fields": {"shard": 3}})
    entry = json.loads(capsys.readouterr().err.strip())
    assert entry["message"] == "done"
    assert entry["shard"] == 3


def test_library_info_logs_are_suppressed(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    logging.getLogger("httpx").info("GET https://example.com")
    assert capsys.readouterr().err == ""
