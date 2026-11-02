from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_atlas.activations.shards import ShardDataset, ShardWriter
from feature_atlas.activations.source import OnlineSource, ShardSource, shuffled_batches
from feature_atlas.models.backend import tiny_backend


def make_dataset(directory: Path) -> ShardDataset:
    """Three shards in which each token's activation records its unique id."""
    writer = ShardWriter(
        directory, "hash", d_model=2, context_size=4, dtype="float32", shard_contexts=5
    )
    token_ids = np.arange(15 * 4, dtype=np.float32).reshape(15, 4)
    writer.add(np.zeros((15, 4)), np.stack([token_ids, token_ids], axis=-1))
    return ShardDataset(directory)


def ids(batches: list[torch.Tensor]) -> list[int]:
    return torch.cat(batches)[:, 0].long().tolist()


def test_one_epoch_serves_every_token_exactly_once(tmp_path: Path) -> None:
    source = ShardSource(make_dataset(tmp_path), seed=0, max_epochs=1)
    assert sorted(ids(list(source.batches(7)))) == list(range(60))


def test_order_is_shuffled_and_repeatable(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    first = ids(list(ShardSource(dataset, seed=0, max_epochs=1).batches(7)))
    again = ids(list(ShardSource(dataset, seed=0, max_epochs=1).batches(7)))
    other = ids(list(ShardSource(dataset, seed=1, max_epochs=1).batches(7)))
    assert first == again
    assert first != sorted(first)
    assert first != other


def test_epochs_use_different_orders(tmp_path: Path) -> None:
    order = ids(list(ShardSource(make_dataset(tmp_path), seed=0, max_epochs=2).batches(7)))
    assert order[:60] != order[60:]
    assert sorted(order[60:]) == list(range(60))


def test_batch_larger_than_a_shard_spans_shards(tmp_path: Path) -> None:
    source = ShardSource(make_dataset(tmp_path), seed=0, max_epochs=1)
    assert next(source.batches(45)).shape == (45, 2)


@pytest.mark.parametrize("stop_after", [1, 3, 8, 9, 12])
def test_resume_is_bit_exact(tmp_path: Path, stop_after: int) -> None:
    dataset = make_dataset(tmp_path)
    uninterrupted = list(ShardSource(dataset, seed=0, max_epochs=2).batches(7))

    first = ShardSource(dataset, seed=0, max_epochs=2)
    served = list(islice(first.batches(7), stop_after))
    state = first.state_dict()

    resumed = ShardSource(dataset, seed=0, max_epochs=2)
    resumed.load_state_dict(state)
    served.extend(resumed.batches(7))

    assert len(served) == len(uninterrupted)
    assert all(torch.equal(a, b) for a, b in zip(served, uninterrupted, strict=True))


def test_empty_cache_is_refused(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, "h", 2, 4, "float32", 1)
    writer.add(np.zeros((1, 4)), np.zeros((1, 4, 2)))
    dataset = ShardDataset(tmp_path)
    dataset.manifest.shards.clear()
    with pytest.raises(ValueError, match="no shards"):
        ShardSource(dataset, seed=0)


def stream(tokens: int, chunk: int = 8) -> list[torch.Tensor]:
    values = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
    return list(values.split(chunk))


def test_buffer_serves_every_token_exactly_once() -> None:
    batches = list(shuffled_batches(stream(1_000), batch_size=16, capacity=64, seed=0))
    assert sorted(ids(batches)) == list(range(1_000))
    assert all(len(batch) == 16 for batch in batches[:-1])


def test_buffer_breaks_up_contexts() -> None:
    # Tokens arrive in contexts of 8. Far more than one context should reach each batch.
    batches = shuffled_batches(stream(4_000), batch_size=32, capacity=512, seed=0)
    contexts_per_batch = [len({int(v) // 8 for v in b[:, 0]}) for b in batches]
    assert np.mean(contexts_per_batch) > 20


def test_buffer_mixes_across_fills() -> None:
    # Half of each fill stays behind, so old tokens must meet tokens from later fills.
    batches = list(shuffled_batches(stream(1_000), batch_size=32, capacity=128, seed=0))
    spreads = [int(batch.max() - batch.min()) for batch in batches[:-1]]
    assert max(spreads) > 128


def test_buffer_is_repeatable() -> None:
    first = list(shuffled_batches(stream(200), 16, 64, seed=3))
    second = list(shuffled_batches(stream(200), 16, 64, seed=3))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_buffer_smaller_than_two_batches_is_refused() -> None:
    with pytest.raises(ValueError, match="capacity"):
        list(shuffled_batches(stream(10), batch_size=16, capacity=20, seed=0))


def test_online_source_flattens_tokens() -> None:
    backend = tiny_backend()
    tokens = [torch.randint(0, 100, (4, 16)) for _ in range(5)]
    source = OnlineSource(backend, tokens, buffer_tokens=64, seed=0)
    batches = list(source.batches(8))
    assert source.d_model == 32
    assert batches[0].shape == (8, 32)
    assert sum(len(batch) for batch in batches) == 5 * 4 * 16
