"""Wiring from a config to concrete data sources. Used by the CLI and notebooks."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from itertools import islice
from pathlib import Path

import torch

from feature_atlas.activations.shards import ShardDataset
from feature_atlas.activations.source import ActivationSource, OnlineSource, ShardSource
from feature_atlas.config import AtlasConfig
from feature_atlas.data.text import Split, iter_contexts, iter_token_batches, load_documents
from feature_atlas.models.backend import ModelBackend, load_backend
from feature_atlas.runtime import artifacts_root


def text_contexts(
    settings: AtlasConfig, backend: ModelBackend, split: Split
) -> Iterator[list[int]]:
    """Stream one split of the configured dataset as fixed-length contexts."""
    return iter_contexts(
        load_documents(settings.data, split),
        backend.encode,
        settings.data.context_size,
        backend.eos_token_id,
        backend.bos_token_id if settings.data.prepend_bos else None,
    )


def build_source(settings: AtlasConfig, device: torch.device) -> ActivationSource:
    """Create the training source the config asks for."""
    source = settings.source
    if source.kind == "shards":
        assert source.shard_dir is not None  # guaranteed by SourceConfig validation
        return ShardSource(
            ShardDataset(source.shard_dir),
            seed=settings.seed,
            exclude_first_position=source.exclude_first_position,
        )
    backend = load_backend(settings.model, device)
    return OnlineSource(
        backend,
        partial(text_contexts, settings, backend, "train"),
        buffer_tokens=source.buffer_tokens,
        seed=settings.seed,
        model_batch_size=source.model_batch_size,
        exclude_first_position=source.exclude_first_position,
    )


def heldout_batches(settings: AtlasConfig, backend: ModelBackend) -> Iterator[torch.Tensor]:
    """The first ``eval.contexts`` held-out contexts, batched for evaluation."""
    contexts = islice(text_contexts(settings, backend, "holdout"), settings.eval.contexts)
    return iter_token_batches(contexts, settings.eval.batch_size)


def run_directory(settings: AtlasConfig) -> Path:
    return artifacts_root() / "runs" / settings.run.name
