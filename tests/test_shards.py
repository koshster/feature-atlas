from pathlib import Path

import numpy as np
import pytest

from feature_atlas.activations.shards import CacheMismatchError, ShardDataset, ShardWriter


def make_writer(
    directory: Path, shard_contexts: int = 4, activation_hash: str = "abc"
) -> ShardWriter:
    return ShardWriter(
        directory,
        activation_hash=activation_hash,
        d_model=3,
        context_size=2,
        dtype="float16",
        shard_contexts=shard_contexts,
    )


def batch(contexts: int, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    tokens = (np.arange(contexts * 2) + start).reshape(contexts, 2)
    return tokens, np.ones((contexts, 2, 3), dtype=np.float32) * (start + 1)


def test_batches_are_split_into_exact_shards(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.add(*batch(3))
    writer.add(*batch(6, start=100))
    writer.flush()
    assert [shard.contexts for shard in writer.manifest.shards] == [4, 4, 1]


def test_round_trip_preserves_order_and_dtypes(tmp_path: Path) -> None:
    writer = make_writer(tmp_path, shard_contexts=2)
    tokens, activations = batch(2)
    writer.add(tokens, activations)
    dataset = ShardDataset(tmp_path)
    shard = dataset.manifest.shards[0]
    assert dataset.tokens(shard).dtype == np.uint16
    assert dataset.activations(shard).dtype == np.float16
    assert np.array_equal(dataset.tokens(shard), tokens)
    assert np.array_equal(dataset.activations(shard), activations)


def test_empty_flush_writes_nothing(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.flush()
    assert writer.manifest.shards == []


def test_nothing_uncommitted_survives_reopening(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.add(*batch(5))  # one shard committed, one context only in memory
    assert make_writer(tmp_path).contexts_committed == 4


def test_reopening_with_other_settings_is_refused(tmp_path: Path) -> None:
    make_writer(tmp_path, shard_contexts=2).add(*batch(2))
    with pytest.raises(CacheMismatchError, match="new directory"):
        make_writer(tmp_path, activation_hash="xyz")


def test_float16_overflow_is_refused(tmp_path: Path) -> None:
    tokens, activations = batch(1)
    with pytest.raises(ValueError, match="Non-finite"):
        make_writer(tmp_path).add(tokens, activations * 1e6)


def test_wrong_shape_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected activations"):
        make_writer(tmp_path).add(np.zeros((1, 2)), np.zeros((1, 2, 5)))


def test_verify_detects_corruption(tmp_path: Path) -> None:
    make_writer(tmp_path, shard_contexts=2).add(*batch(2))
    dataset = ShardDataset(tmp_path)
    dataset.verify()
    path = tmp_path / dataset.manifest.shards[0].tokens_name
    path.write_bytes(path.read_bytes()[:-1] + b"\x7f")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        dataset.verify()


def test_low_disk_stops_before_writing(tmp_path: Path) -> None:
    writer = make_writer(tmp_path, shard_contexts=2)
    writer.min_free_gb = 10**9
    with pytest.raises(OSError, match="Stopping before filling the disk"):
        writer.add(*batch(2))
    assert not list(tmp_path.glob("*.npy"))


def test_missing_cache_gives_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No activation cache"):
        ShardDataset(tmp_path)
