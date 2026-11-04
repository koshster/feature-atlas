from collections.abc import Iterator
from itertools import count, islice
from pathlib import Path

import pytest
import torch

from feature_atlas.activations.source import TensorSource
from feature_atlas.config import AtlasConfig, RunConfig, SAEConfig, TrainConfig
from feature_atlas.sae.checkpoint import CheckpointManager, CheckpointMismatchError
from feature_atlas.sae.model import SparseAutoencoder
from feature_atlas.sae.train import l1_coefficient, learning_rate, train_sae
from feature_atlas.tracking import JsonlTracker, NullTracker, read_metrics
from tests.synthetic import mean_max_cosine, sparse_dictionary_data

CPU = torch.device("cpu")


def make_config(**train: object) -> AtlasConfig:
    settings = {
        "total_tokens": 256 * 1500,
        "batch_size": 256,
        "lr": 3e-3,
        "l1_coefficient": 1.0,
        "lr_warmup_steps": 50,
        "init_tokens": 2048,
        "dead_window_steps": 200,
        "resample_every_steps": 300,
        "resample_tokens": 4096,
    }
    settings.update(train)
    return AtlasConfig(
        sae=SAEConfig(d_sae=64),
        train=TrainConfig.model_validate(settings),
        run=RunConfig(log_every_steps=100),
    )


def test_sae_recovers_a_planted_dictionary(tmp_path: Path) -> None:
    # The most important test: if the SAE can't find directions we planted, nothing it
    # finds in GPT-2 can be trusted.
    data, planted = sparse_dictionary_data()
    untrained = mean_max_cosine(SparseAutoencoder(32, 64).W_dec.detach(), planted)
    sae = train_sae(make_config(), TensorSource(data, seed=0), tmp_path, NullTracker(), CPU)
    recovered = mean_max_cosine(sae.W_dec.detach(), planted)
    assert untrained.mean() < 0.7  # control: random directions don't match by chance
    assert recovered.min() > 0.95
    with torch.no_grad():
        l0 = (sae(data[:5000]).features > 0).sum(dim=1).float().mean()
    assert 2 < l0 < 8  # about 2.9 features are active per token in the data


class KilledSource(TensorSource):
    """Raises partway through, like a GPU session that dies."""

    def __init__(self, data: torch.Tensor, seed: int, die_after: int) -> None:
        super().__init__(data, seed)
        self.die_after = die_after

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        for served, batch in zip(count(), super().batches(batch_size), strict=False):
            if served == self.die_after:
                raise RuntimeError("session killed")
            yield batch


def test_killed_run_resumes_to_identical_weights(tmp_path: Path) -> None:
    data, _ = sparse_dictionary_data(tokens=5_000)
    # Short windows so resampling happens on both sides of the kill.
    config = make_config(
        total_tokens=256 * 60, dead_window_steps=5, resample_every_steps=7, resample_tokens=512
    ).model_copy(update={"run": RunConfig(checkpoint_every_steps=10, log_every_steps=5)})

    clean = train_sae(config, TensorSource(data, 0), tmp_path / "clean", NullTracker(), CPU)

    with pytest.raises(RuntimeError, match="session killed"):
        train_sae(
            config, KilledSource(data, 0, die_after=40), tmp_path / "killed", NullTracker(), CPU
        )
    latest = CheckpointManager(tmp_path / "killed" / "checkpoints", config.training_hash()).latest()
    assert latest is not None and latest.name == "step_000000020.pt"

    resumed = train_sae(config, TensorSource(data, 0), tmp_path / "killed", NullTracker(), CPU)
    for name, tensor in clean.state_dict().items():
        assert torch.equal(tensor, resumed.state_dict()[name]), name


def test_changed_config_refuses_to_resume(tmp_path: Path) -> None:
    data, _ = sparse_dictionary_data(tokens=2_000)
    train_sae(
        make_config(total_tokens=256 * 5), TensorSource(data, 0), tmp_path, NullTracker(), CPU
    )
    with pytest.raises(CheckpointMismatchError, match="new run name"):
        train_sae(
            make_config(total_tokens=256 * 5, lr=5e-4),
            TensorSource(data, 0),
            tmp_path,
            NullTracker(),
            CPU,
        )


def test_time_based_checkpoints_and_pruning(tmp_path: Path) -> None:
    data, _ = sparse_dictionary_data(tokens=2_000)
    ticks = count()
    config = make_config(total_tokens=256 * 30).model_copy(
        update={"run": RunConfig(checkpoint_every_minutes=1, keep_checkpoints=2)}
    )
    # Each clock read advances 7 s, so a minute passes every few steps.
    train_sae(
        config, TensorSource(data, 0), tmp_path, NullTracker(), CPU, clock=lambda: next(ticks) * 7.0
    )
    saved = CheckpointManager(tmp_path / "checkpoints", config.training_hash()).all()
    assert len(saved) == 2
    assert saved[-1].name == "step_000000030.pt"


def test_metrics_are_logged(tmp_path: Path) -> None:
    data, _ = sparse_dictionary_data(tokens=5_000)
    tracker = JsonlTracker(tmp_path / "metrics.jsonl")
    train_sae(make_config(total_tokens=256 * 300), TensorSource(data, 0), tmp_path, tracker, CPU)
    tracker.close()
    metrics = read_metrics(tmp_path / "metrics.jsonl")
    assert [entry["step"] for entry in metrics] == [100, 200, 300]
    last = metrics[-1]
    assert 0 < last["explained_variance"] <= 1
    assert last["l0"] > 0
    assert 0 <= last["dead_fraction"] <= 1


def test_source_running_dry_is_reported(tmp_path: Path) -> None:
    class Finite(TensorSource):
        def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
            yield from islice(super().batches(batch_size), 12)

    data, _ = sparse_dictionary_data(tokens=2_000)
    with pytest.raises(RuntimeError, match="ran out"):
        train_sae(make_config(), Finite(data, 0), tmp_path, NullTracker(), CPU)


def test_learning_rate_schedule() -> None:
    config = TrainConfig(total_tokens=1000, batch_size=10, lr=1.0, lr_warmup_steps=10)
    assert learning_rate(0, config) == pytest.approx(0.1)
    assert learning_rate(50, config) == 1.0
    assert learning_rate(90, config) == pytest.approx(0.5)
    assert learning_rate(99, config) == pytest.approx(0.05)


def test_l1_warmup() -> None:
    config = TrainConfig(
        total_tokens=1000, batch_size=10, l1_coefficient=2.0, l1_warmup_fraction=0.1
    )
    assert l1_coefficient(0, config) == pytest.approx(0.2)
    assert l1_coefficient(50, config) == 2.0
