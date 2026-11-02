from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_atlas.activations.cache import build_cache
from feature_atlas.activations.shards import ShardDataset
from feature_atlas.config import AtlasConfig, CacheConfig, DataConfig, ModelConfig
from feature_atlas.models.backend import tiny_backend


def settings(total: int = 12) -> AtlasConfig:
    return AtlasConfig(
        model=ModelConfig(hook_layer=1),
        data=DataConfig(context_size=16),
        cache=CacheConfig(total_contexts=total, shard_contexts=4, batch_size=3, min_free_gb=0),
    )


def contexts(count: int) -> Iterator[list[int]]:
    generator = torch.Generator().manual_seed(0)
    for _ in range(count):
        yield torch.randint(0, 100, (16,), generator=generator).tolist()


def read_all(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = ShardDataset(directory)
    shards = dataset.manifest.shards
    return (
        np.concatenate([dataset.tokens(shard) for shard in shards]),
        np.concatenate([dataset.activations(shard) for shard in shards]),
    )


def test_cached_values_match_the_model(tmp_path: Path) -> None:
    backend = tiny_backend()
    build_cache(settings(), backend, contexts(100), tmp_path)
    tokens, activations = read_all(tmp_path)
    assert tokens.shape == (12, 16)
    assert tokens.tolist() == list(contexts(12))
    expected = backend.activations(torch.tensor(tokens.astype(np.int64))).numpy()
    assert np.array_equal(activations, expected.astype(np.float16))


def test_interrupted_cache_resumes_to_identical_bytes(tmp_path: Path) -> None:
    def dying_stream() -> Iterator[list[int]]:
        for index, context in enumerate(contexts(100)):
            if index == 7:
                raise RuntimeError("session killed")
            yield context

    backend = tiny_backend()
    with pytest.raises(RuntimeError, match="session killed"):
        build_cache(settings(), backend, dying_stream(), tmp_path / "resumed")
    assert ShardDataset(tmp_path / "resumed").manifest.total_contexts == 4

    build_cache(settings(), backend, contexts(100), tmp_path / "resumed")
    build_cache(settings(), backend, contexts(100), tmp_path / "clean")

    for resumed, clean in zip(
        read_all(tmp_path / "resumed"), read_all(tmp_path / "clean"), strict=True
    ):
        assert np.array_equal(resumed, clean)


def test_rerunning_a_finished_cache_does_nothing(tmp_path: Path) -> None:
    backend = tiny_backend()
    first = build_cache(settings(), backend, contexts(100), tmp_path)
    second = build_cache(settings(), backend, contexts(100), tmp_path)
    assert first == second


def test_cache_can_be_extended(tmp_path: Path) -> None:
    backend = tiny_backend()
    build_cache(settings(total=8), backend, contexts(100), tmp_path)
    manifest = build_cache(settings(total=12), backend, contexts(100), tmp_path)
    assert manifest.total_contexts == 12
    assert read_all(tmp_path)[0].tolist() == list(contexts(12))


def test_short_stream_warns_and_keeps_what_it_has(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    manifest = build_cache(settings(), tiny_backend(), contexts(5), tmp_path)
    assert manifest.total_contexts == 5
    assert "text stream ended early" in caplog.text
