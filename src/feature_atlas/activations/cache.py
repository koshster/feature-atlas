"""Build an activation cache from a stream of contexts."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from feature_atlas.activations.shards import Manifest, ShardWriter
from feature_atlas.config import AtlasConfig
from feature_atlas.data.text import iter_token_batches
from feature_atlas.models.backend import ModelBackend

logger = logging.getLogger(__name__)


def build_cache(
    config: AtlasConfig,
    backend: ModelBackend,
    contexts: Iterable[list[int]],
    directory: Path,
) -> Manifest:
    """Cache ``config.cache.total_contexts`` contexts, resuming any earlier run.

    ``contexts`` must be the same deterministic stream on every attempt. A resumed run
    skips the contexts already committed without running them through the model.
    """
    writer = ShardWriter(
        directory,
        activation_hash=config.activation_hash(),
        d_model=backend.d_model,
        context_size=config.data.context_size,
        dtype=config.cache.dtype,
        shard_contexts=config.cache.shard_contexts,
        min_free_gb=config.cache.min_free_gb,
    )
    remaining = islice(contexts, writer.contexts_committed, config.cache.total_contexts)
    for tokens in iter_token_batches(remaining, config.cache.batch_size):
        activations = backend.activations(tokens)  # [batch, context, d_model]
        writer.add(tokens.numpy(), activations.cpu().numpy())
    writer.flush()
    if writer.contexts_committed < config.cache.total_contexts:
        logger.warning(
            "text stream ended early",
            extra={
                "fields": {
                    "contexts": writer.contexts_committed,
                    "requested": config.cache.total_contexts,
                }
            },
        )
    return writer.manifest
