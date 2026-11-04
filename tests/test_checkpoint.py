from pathlib import Path

import pytest
import torch

from feature_atlas.sae.checkpoint import CheckpointManager, CheckpointMismatchError


def test_save_load_round_trip(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, "abc")
    generator = torch.Generator().manual_seed(3)
    manager.save(5, {"weights": torch.ones(3), "generator": generator.get_state()})
    payload = manager.load(manager.latest())  # type: ignore[arg-type]
    assert payload["step"] == 5
    assert torch.equal(payload["weights"], torch.ones(3))
    restored = torch.Generator()
    restored.set_state(payload["generator"])
    assert torch.equal(torch.rand(2, generator=restored), torch.rand(2, generator=generator))


def test_only_the_newest_are_kept(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, "abc", keep=2)
    for step in (10, 20, 30):
        manager.save(step, {})
    assert [path.name for path in manager.all()] == ["step_000000020.pt", "step_000000030.pt"]


def test_ignores_unrelated_and_temporary_files(tmp_path: Path) -> None:
    (tmp_path / "step_000000099.pt.tmp").write_bytes(b"partial")
    (tmp_path / "notes.txt").write_text("hi")
    assert CheckpointManager(tmp_path, "abc").latest() is None


def test_other_config_is_refused(tmp_path: Path) -> None:
    CheckpointManager(tmp_path, "abc").save(1, {})
    with pytest.raises(CheckpointMismatchError, match="new run name"):
        CheckpointManager(tmp_path, "xyz").load(tmp_path / "step_000000001.pt")
