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


class SourceConfig(_Frozen):
    """Where training activations come from, and which positions are used."""

    kind: Literal["online", "shards"] = "online"
    shard_dir: Path | None = None
    # Position 0 has about 30x the norm of other positions in GPT-2, whatever the token,
    # and would dominate an L2 loss. See the Phase 1 entry in docs/PROGRESS.md.
    exclude_first_position: bool = True
    # 2**19 tokens is about 1.6 GB at d_model 768 in float32.
    buffer_tokens: int = Field(default=2**19, ge=2)
    model_batch_size: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def _shards_need_a_directory(self) -> SourceConfig:
        if self.kind == "shards" and self.shard_dir is None:
            raise ValueError("source.kind 'shards' needs source.shard_dir")
        return self


class SAEConfig(_Frozen):
    """Dictionary shape. The input width comes from the model."""

    d_sae: int = Field(default=24_576, ge=1)  # 32 x 768


class TrainConfig(_Frozen):
    """Optimization settings. Defaults are starting points for the Phase 3 sweep."""

    total_tokens: int = Field(default=300_000_000, ge=1)
    batch_size: int = Field(default=4096, ge=1)
    lr: float = Field(default=5e-5, gt=0)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_warmup_steps: int = Field(default=1_000, ge=0)
    # The final fraction of steps decays the learning rate linearly to zero.
    lr_decay_fraction: float = Field(default=0.2, ge=0, le=1)
    # λ, in units where inputs are scaled to mean squared norm d_model. Not comparable
    # with other implementations' λ; compare at matched L0 instead.
    l1_coefficient: float = Field(default=5.0, ge=0)
    l1_warmup_fraction: float = Field(default=0.05, ge=0, le=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    # Tokens used once, before training, to set the input scale and b_dec.
    init_tokens: int = Field(default=2**16, ge=1)
    dead_window_steps: int = Field(default=12_500, ge=1)
    resample_every_steps: int = Field(default=25_000, ge=1)
    resample_until_fraction: float = Field(default=0.5, ge=0, le=1)
    resample_tokens: int = Field(default=2**17, ge=1)
    # Features firing on more than this fraction of tokens are reported as dense.
    dense_threshold: float = Field(default=0.1, gt=0, le=1)

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.batch_size


class RunConfig(_Frozen):
    """Bookkeeping that doesn't change results, so it's excluded from the training hash."""

    name: str = "default"
    checkpoint_every_minutes: float = Field(default=10.0, gt=0)
    checkpoint_every_steps: int | None = Field(default=None, ge=1)
    keep_checkpoints: int = Field(default=2, ge=1)
    log_every_steps: int = Field(default=100, ge=1)


class EvalConfig(_Frozen):
    """Held-out evaluation size."""

    contexts: int = Field(default=512, ge=1)
    batch_size: int = Field(default=16, ge=1)


class AtlasConfig(_Frozen):
    """Top-level experiment config."""

    seed: int = 0
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    cache: CacheConfig = CacheConfig()
    source: SourceConfig = SourceConfig()
    sae: SAEConfig = SAEConfig()
    train: TrainConfig = TrainConfig()
    run: RunConfig = RunConfig()
    eval: EvalConfig = EvalConfig()

    def training_hash(self) -> str:
        """Hash everything that determines a trained SAE.

        A checkpoint is only resumed under the same hash. The shard directory is left out
        because the same cache sits at different paths on a laptop and on Kaggle.
        """
        return stable_hash(
            {
                "seed": self.seed,
                "model": self.model.model_dump(),
                "data": self.data.model_dump(),
                "source": self.source.model_dump(mode="json", exclude={"shard_dir"}),
                "sae": self.sae.model_dump(),
                "train": self.train.model_dump(),
            }
        )

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
