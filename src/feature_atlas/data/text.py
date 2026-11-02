"""Stream documents and turn them into fixed-length token contexts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import Literal

import torch

from feature_atlas.config import DataConfig

Split = Literal["train", "holdout"]


def assign_split(text: str, holdout_fraction: float, salt: str) -> Split:
    """Assign a document to a split using only its content.

    A streaming dataset has no random access, so an index-based split isn't possible.
    Hashing the text gives the same answer on every machine and in any order, and
    duplicate documents always land together, which prevents train/holdout leakage.
    """
    digest = hashlib.blake2b(f"{salt}:{text}".encode(), digest_size=8).digest()
    position = int.from_bytes(digest, "big") / 2**64  # uniform in [0, 1)
    return "holdout" if position < holdout_fraction else "train"


def load_documents(config: DataConfig, split: Split) -> Iterator[str]:
    """Stream one split's non-empty documents from a Hugging Face dataset."""
    from datasets import load_dataset

    stream = load_dataset(config.dataset, split=config.dataset_split, streaming=True)
    for row in stream:
        text = row[config.text_column]
        if text and assign_split(text, config.holdout_fraction, config.split_salt) == split:
            yield text


def iter_contexts(
    documents: Iterable[str],
    encode: Callable[[str], list[int]],
    context_size: int,
    eos_token_id: int,
    bos_token_id: int | None = None,
) -> Iterator[list[int]]:
    """Concatenate documents into contexts of exactly ``context_size`` tokens.

    Documents are joined with EOS, so short ones aren't wasted on padding and long ones
    span several contexts. Tokens left over at the end of the stream are dropped.
    """
    prefix = [] if bos_token_id is None else [bos_token_id]
    body_size = context_size - len(prefix)
    pending: list[int] = []
    for document in documents:
        pending.extend(encode(document))
        pending.append(eos_token_id)
        offset = 0
        while len(pending) - offset >= body_size:
            yield prefix + pending[offset : offset + body_size]
            offset += body_size
        del pending[:offset]


def iter_token_batches(contexts: Iterable[list[int]], batch_size: int) -> Iterator[torch.Tensor]:
    """Group contexts into ``[batch, context]`` tensors; the last may be smaller."""
    iterator = iter(contexts)
    while batch := list(islice(iterator, batch_size)):
        yield torch.tensor(batch, dtype=torch.long)
