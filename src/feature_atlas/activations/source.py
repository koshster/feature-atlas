"""Sources of shuffled activation batches for training and evaluation.

An SAE sees individual tokens. Tokens from one context are highly correlated, so feeding
contexts in order would make each batch a sample of very few documents. Both sources
therefore mix tokens before batching. The trainer depends only on ``ActivationSource``,
so it can't tell live activations from cached ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

import numpy as np
import torch

from feature_atlas.activations.shards import ShardDataset, ShardInfo
from feature_atlas.models.backend import ModelBackend


class ActivationSource(Protocol):
    """Yields ``[batch, d_model]`` float32 tensors of shuffled tokens."""

    @property
    def d_model(self) -> int: ...

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]: ...


class ShardSource:
    """Serves cached activations, with bit-exact resume.

    Each epoch's shard order and each shard's token order are pure functions of
    ``(seed, epoch, shard)``. Resuming therefore needs only the epoch and a token count
    rather than any saved generator state.

    Shuffling is within a shard, not across the whole cache, because random reads across
    files would be far slower. Shards should hold many documents for this reason.
    """

    def __init__(self, dataset: ShardDataset, seed: int, max_epochs: int | None = None) -> None:
        if not dataset.manifest.shards:
            raise ValueError(f"Cache at {dataset.directory} has no shards")
        self.dataset = dataset
        self.seed = seed
        self.max_epochs = max_epochs
        self.epoch = 0
        self.tokens_into_epoch = 0

    @property
    def d_model(self) -> int:
        return self.dataset.manifest.d_model

    def state_dict(self) -> dict[str, int]:
        """Return the position of the next unserved token."""
        return {"epoch": self.epoch, "tokens_into_epoch": self.tokens_into_epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = state["epoch"]
        self.tokens_into_epoch = state["tokens_into_epoch"]

    def _shard_order(self) -> list[ShardInfo]:
        shards = self.dataset.manifest.shards
        order = np.random.default_rng([self.seed, self.epoch]).permutation(len(shards))
        return [shards[int(index)] for index in order]

    def _tokens_in(self, shard: ShardInfo) -> int:
        return shard.contexts * self.dataset.manifest.context_size

    def _locate(self) -> tuple[ShardInfo, int] | None:
        """Find the shard and offset of the next token, or ``None`` at the epoch's end."""
        remaining = self.tokens_into_epoch
        for shard in self._shard_order():
            size = self._tokens_in(shard)
            if remaining < size:
                return shard, remaining
            remaining -= size
        return None

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        """Yield batches until ``max_epochs``, or forever if it is ``None``.

        The position advances before each ``yield``, so ``state_dict`` taken after
        receiving a batch resumes at the following batch.
        """
        pending: list[np.ndarray] = []
        pending_tokens = 0
        while self.max_epochs is None or self.epoch < self.max_epochs:
            located = self._locate()
            if located is None:
                self.epoch += 1
                self.tokens_into_epoch = 0
                continue
            shard, offset = located
            order = np.random.default_rng([self.seed, self.epoch, shard.index]).permutation(
                self._tokens_in(shard)
            )
            flat = self.dataset.activations(shard).reshape(-1, self.d_model)
            while offset < len(order):
                take = min(batch_size - pending_tokens, len(order) - offset)
                pending.append(np.asarray(flat[order[offset : offset + take]], dtype=np.float32))
                pending_tokens += take
                offset += take
                self.tokens_into_epoch += take
                if pending_tokens == batch_size:
                    yield torch.from_numpy(np.concatenate(pending))
                    pending, pending_tokens = [], 0
        if pending:
            yield torch.from_numpy(np.concatenate(pending))


def shuffled_batches(
    chunks: Iterable[torch.Tensor], batch_size: int, capacity: int, seed: int
) -> Iterator[torch.Tensor]:
    """Mix a stream of ``[tokens, d_model]`` chunks through a shuffle buffer.

    When the buffer fills it is permuted, half is served, and the other half stays to mix
    with incoming tokens. Keeping half is what lets tokens from different fills meet: a
    plain fill-and-drain could never put tokens from two fills in the same batch.
    """
    if capacity < 2 * batch_size:
        raise ValueError(f"capacity {capacity} must be at least twice batch_size {batch_size}")
    generator = torch.Generator().manual_seed(seed)
    served_per_fill = (capacity // 2 // batch_size) * batch_size
    buffer: torch.Tensor | None = None
    filled = 0
    for chunk in chunks:
        if buffer is None:
            buffer = torch.empty(capacity, chunk.shape[1], dtype=chunk.dtype, device=chunk.device)
        start = 0
        while start < len(chunk):
            take = min(capacity - filled, len(chunk) - start)
            buffer[filled : filled + take] = chunk[start : start + take]
            filled += take
            start += take
            if filled == capacity:
                buffer = buffer[torch.randperm(capacity, generator=generator).to(buffer.device)]
                for begin in range(0, served_per_fill, batch_size):
                    yield buffer[begin : begin + batch_size].clone()
                buffer[: capacity - served_per_fill] = buffer[served_per_fill:].clone()
                filled = capacity - served_per_fill
    if buffer is not None and filled:
        remainder = buffer[:filled][torch.randperm(filled, generator=generator).to(buffer.device)]
        for begin in range(0, filled, batch_size):
            yield remainder[begin : begin + batch_size]


class OnlineSource:
    """Generates activations live from the model, trading GPU time for disk.

    Resume is statistical, not bit-exact: the buffer's contents aren't saved. See
    ``docs/PLAN.md`` for why that is an acceptable tradeoff.
    """

    def __init__(
        self,
        backend: ModelBackend,
        token_batches: Iterable[torch.Tensor],
        buffer_tokens: int,
        seed: int,
    ) -> None:
        self.backend = backend
        self.token_batches = token_batches
        self.buffer_tokens = buffer_tokens
        self.seed = seed

    @property
    def d_model(self) -> int:
        return self.backend.d_model

    def _chunks(self) -> Iterator[torch.Tensor]:
        for tokens in self.token_batches:
            yield self.backend.activations(tokens).flatten(0, 1)

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        """Yield shuffled batches until the token stream ends."""
        return shuffled_batches(self._chunks(), batch_size, self.buffer_tokens, self.seed)
