"""Sources of shuffled activation batches for training and evaluation.

An SAE sees individual tokens. Tokens from one context are highly correlated, so feeding
contexts in order would make each batch a sample of very few documents. Every source
therefore mixes tokens before batching. The trainer depends only on ``ActivationSource``,
so it can't tell live activations from cached or synthetic ones.

Sources skip position 0 by default. GPT-2's first position has about 30x the norm of
the rest, whatever the token, and would dominate an L2 reconstruction loss.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import Protocol

import numpy as np
import torch

from feature_atlas.activations.shards import ShardDataset, ShardInfo
from feature_atlas.data.text import iter_token_batches
from feature_atlas.models.backend import ModelBackend


class ActivationSource(Protocol):
    """Yields ``[batch, d_model]`` float32 tensors of shuffled tokens.

    ``state_dict`` taken between batches must let a fresh source continue from the
    next batch. Whether that continuation is exact depends on the source.
    """

    @property
    def d_model(self) -> int: ...

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]: ...

    def state_dict(self) -> dict[str, int]: ...

    def load_state_dict(self, state: dict[str, int]) -> None: ...


def _epoch_permutation(seed: int, epoch: int, size: int, *keys: int) -> np.ndarray:
    return np.random.default_rng([seed, epoch, *keys]).permutation(size)


class TensorSource:
    """Serves an in-memory ``[tokens, d_model]`` tensor, reshuffled every epoch.

    Used for tests and synthetic experiments. Resume is exact, for the same reason as
    ``ShardSource``: each epoch's order is a pure function of ``(seed, epoch)``.
    """

    def __init__(self, data: torch.Tensor, seed: int) -> None:
        self.data = data
        self.seed = seed
        self.epoch = 0
        self.offset = 0

    @property
    def d_model(self) -> int:
        return int(self.data.shape[1])

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch, self.offset = state["epoch"], state["offset"]

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        """Yield batches forever; a batch never spans two epochs."""
        while True:
            order = torch.from_numpy(_epoch_permutation(self.seed, self.epoch, len(self.data)))
            while self.offset + batch_size <= len(order):
                indices = order[self.offset : self.offset + batch_size]
                self.offset += batch_size
                yield self.data[indices]
            self.epoch += 1
            self.offset = 0


class ShardSource:
    """Serves cached activations, with bit-exact resume.

    Each epoch's shard order and each shard's token order are pure functions of
    ``(seed, epoch, shard)``. Resuming therefore needs only the epoch and a token count
    rather than any saved generator state.

    Shuffling is within a shard, not across the whole cache, because random reads across
    files would be far slower. Shards should hold many documents for this reason.
    """

    def __init__(
        self,
        dataset: ShardDataset,
        seed: int,
        max_epochs: int | None = None,
        exclude_first_position: bool = True,
    ) -> None:
        if not dataset.manifest.shards:
            raise ValueError(f"Cache at {dataset.directory} has no shards")
        self.dataset = dataset
        self.seed = seed
        self.max_epochs = max_epochs
        self.first_position = 1 if exclude_first_position else 0
        self.epoch = 0
        self.tokens_into_epoch = 0

    @property
    def d_model(self) -> int:
        return self.dataset.manifest.d_model

    @property
    def positions_per_context(self) -> int:
        return self.dataset.manifest.context_size - self.first_position

    def state_dict(self) -> dict[str, int]:
        """Return the position of the next unserved token."""
        return {"epoch": self.epoch, "tokens_into_epoch": self.tokens_into_epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = state["epoch"]
        self.tokens_into_epoch = state["tokens_into_epoch"]

    def _shard_order(self) -> list[ShardInfo]:
        shards = self.dataset.manifest.shards
        order = _epoch_permutation(self.seed, self.epoch, len(shards))
        return [shards[int(index)] for index in order]

    def _tokens_in(self, shard: ShardInfo) -> int:
        return shard.contexts * self.positions_per_context

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
            order = _epoch_permutation(self.seed, self.epoch, self._tokens_in(shard), shard.index)
            activations = self.dataset.activations(shard)  # [contexts, context_size, d_model]
            while offset < len(order):
                take = min(batch_size - pending_tokens, len(order) - offset)
                # A token index counts kept positions only, so map it back to the array.
                chosen = order[offset : offset + take]
                contexts = chosen // self.positions_per_context
                positions = chosen % self.positions_per_context + self.first_position
                pending.append(np.asarray(activations[contexts, positions], dtype=np.float32))
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

    Resume is statistical, not bit-exact. The state records how many contexts entered the
    buffer, and a resumed source skips them, so tokens that were in the buffer but not yet
    served are lost: at most ``buffer_tokens`` plus one model batch, because a batch
    counts as consumed once it has been through the model. Saving the buffer itself
    would cost about 1.6 GB per checkpoint.

    ``contexts`` is a factory rather than an iterator so that each call to ``batches``
    starts a fresh, deterministic stream that can be skipped forward.
    """

    def __init__(
        self,
        backend: ModelBackend,
        contexts: Callable[[], Iterable[list[int]]],
        buffer_tokens: int,
        seed: int,
        model_batch_size: int = 32,
        exclude_first_position: bool = True,
    ) -> None:
        self.backend = backend
        self.contexts = contexts
        self.buffer_tokens = buffer_tokens
        self.seed = seed
        self.model_batch_size = model_batch_size
        self.first_position = 1 if exclude_first_position else 0
        self.contexts_consumed = 0

    @property
    def d_model(self) -> int:
        return self.backend.d_model

    def state_dict(self) -> dict[str, int]:
        return {"contexts_consumed": self.contexts_consumed}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.contexts_consumed = state["contexts_consumed"]

    def _chunks(self) -> Iterator[torch.Tensor]:
        remaining = islice(self.contexts(), self.contexts_consumed, None)
        for tokens in iter_token_batches(remaining, self.model_batch_size):
            activations = self.backend.activations(tokens)  # [batch, context, d_model]
            self.contexts_consumed += len(tokens)
            yield activations[:, self.first_position :].flatten(0, 1)

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        """Yield shuffled batches until the context stream ends."""
        # Mixing in the position stops a resumed run from replaying the same shuffles.
        seed = int(np.random.SeedSequence([self.seed, self.contexts_consumed]).generate_state(1)[0])
        return shuffled_batches(self._chunks(), batch_size, self.buffer_tokens, seed)
