"""Validated experiment configuration, loaded from YAML.

A config file plus a seed is the source of truth for every result, so unknown
keys are rejected: a typo must fail loudly instead of silently using a default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Frozen):
    """Which model to hook, and the settings that change its activations.

    The processing flags default to TransformerLens's defaults, which are also the
    settings the reference ``gpt2-small-res-jb`` SAEs were trained with. Changing any
    of them changes the activations, so they are part of the activation hash.
    """

    name: str = "gpt2"
    hook_layer: int = Field(default=8, ge=0)
    fold_ln: bool = True
    center_writing_weights: bool = True
    center_unembed: bool = True
    fold_value_biases: bool = True

    @property
    def hook_point(self) -> str:
        return f"blocks.{self.hook_layer}.hook_resid_pre"


class DataConfig(_Frozen):
    """Where text comes from and how documents become fixed-length contexts."""

    dataset: str = "NeelNanda/pile-10k"
    dataset_split: str = "train"
    text_column: str = "text"
    context_size: int = Field(default=128, ge=2)
    prepend_bos: bool = True
    holdout_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    # Changing the salt redraws the train/holdout split.
    split_salt: str = "feature-atlas-v1"


class CacheConfig(_Frozen):
    """How many activations to cache, and how to lay them out on disk."""

    total_contexts: int = Field(default=2_000, ge=1)
    shard_contexts: int = Field(default=500, ge=1)
    batch_size: int = Field(default=32, ge=1)
    dtype: Literal["float16", "float32"] = "float16"
    # Refuse to start a shard that would leave less than this on the disk.
    min_free_gb: float = Field(default=2.0, ge=0.0)

    @model_validator(mode="after")
    def _shard_fits(self) -> CacheConfig:
        if self.shard_contexts > self.total_contexts:
            raise ValueError("shard_contexts cannot exceed total_contexts")
        return self


class AtlasConfig(_Frozen):
    """Top-level experiment config."""

    seed: int = 0
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    cache: CacheConfig = CacheConfig()

    def activation_hash(self) -> str:
        """Hash everything that determines the cached bytes.

        ``total_contexts`` is left out so a cache can be extended, and ``batch_size``
        and ``min_free_gb`` because they affect speed and safety, not the values.
        """
        return stable_hash(
            {
                "seed": self.seed,
                "model": self.model.model_dump(),
                "data": self.data.model_dump(),
                "shard_contexts": self.cache.shard_contexts,
                "dtype": self.cache.dtype,
            }
        )


def stable_hash(payload: dict[str, Any]) -> str:
    """Return a short hash of JSON-compatible data, independent of key order."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_config(path: Path) -> AtlasConfig:
    """Load and validate a YAML config, raising ``ValueError`` with the file named."""
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as error:
        raise ValueError(f"Cannot read config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    try:
        return AtlasConfig.model_validate(raw or {})
    except ValidationError as error:
        raise ValueError(f"Invalid config {path}:\n{error}") from error
