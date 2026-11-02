"""Activation shards on disk.

Each shard is a pair of ``.npy`` files, chosen over safetensors because ``numpy.memmap``
gives random access without loading a whole shard:

- ``shard_NNNNN.activations.npy``: ``[contexts, context_size, d_model]``, float16 or 32
- ``shard_NNNNN.tokens.npy``: ``[contexts, context_size]``, uint16

``manifest.json`` is rewritten atomically after each shard, so it is the commit record.
A kill between the two steps leaves an orphan shard that the next run overwrites.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
_BYTES_PER_GB = 1024**3


class CacheMismatchError(ValueError):
    """An existing cache was built with different settings."""


class ShardInfo(BaseModel):
    """One committed shard and the checksums of its files."""

    index: int
    contexts: int
    activations_sha256: str
    tokens_sha256: str

    @property
    def activations_name(self) -> str:
        return f"shard_{self.index:05d}.activations.npy"

    @property
    def tokens_name(self) -> str:
        return f"shard_{self.index:05d}.tokens.npy"


class Manifest(BaseModel):
    """Everything needed to interpret and validate a cache directory."""

    activation_hash: str
    d_model: int
    context_size: int
    dtype: str
    shards: list[ShardInfo] = []

    @property
    def total_contexts(self) -> int:
        return sum(shard.contexts for shard in self.shards)


def _sync_and_replace(temporary: Path, target: Path) -> None:
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _atomic_save_array(path: Path, array: NDArray[Any]) -> None:
    # Rename is atomic on one filesystem, so readers never see a half-written file.
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    _sync_and_replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(directory: Path) -> Manifest:
    """Read a cache directory's manifest."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"No activation cache at {directory}: {MANIFEST_NAME} is missing")
    return Manifest.model_validate_json(path.read_text())


class ShardWriter:
    """Buffers activations and commits them as fixed-size shards.

    Memory use is bounded by one shard plus one incoming batch, however large the cache.
    Reopening a directory continues after its last committed shard.
    """

    def __init__(
        self,
        directory: Path,
        activation_hash: str,
        d_model: int,
        context_size: int,
        dtype: str,
        shard_contexts: int,
        min_free_gb: float = 0.0,
    ) -> None:
        self.directory = directory
        self.shard_contexts = shard_contexts
        self.min_free_gb = min_free_gb
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / MANIFEST_NAME).exists():
            self.manifest = read_manifest(directory)
            if self.manifest.activation_hash != activation_hash:
                raise CacheMismatchError(
                    f"{directory} holds activations with hash {self.manifest.activation_hash}, "
                    f"but this config has {activation_hash}. Use a new directory."
                )
        else:
            self.manifest = Manifest(
                activation_hash=activation_hash,
                d_model=d_model,
                context_size=context_size,
                dtype=dtype,
            )
        self._tokens: list[NDArray[np.uint16]] = []
        self._activations: list[NDArray[Any]] = []
        self._buffered = 0

    @property
    def contexts_committed(self) -> int:
        return self.manifest.total_contexts

    def add(self, tokens: NDArray[Any], activations: NDArray[Any]) -> None:
        """Buffer one batch: ``[batch, context]`` tokens, ``[batch, context, d_model]`` values."""
        expected = (self.manifest.context_size, self.manifest.d_model)
        if activations.shape[1:] != expected:
            raise ValueError(f"Expected activations [batch, {expected}], got {activations.shape}")
        # float16 tops out near 65,504, so an overflow would silently store inf.
        with np.errstate(over="ignore"):
            narrowed = activations.astype(self.manifest.dtype)
        if not np.isfinite(narrowed).all():
            raise ValueError(f"Non-finite activations after converting to {self.manifest.dtype}")
        self._tokens.append(tokens.astype(np.uint16))
        self._activations.append(narrowed)
        self._buffered += len(tokens)
        while self._buffered >= self.shard_contexts:
            self._commit(self.shard_contexts)

    def flush(self) -> None:
        """Commit any buffered contexts as a final, possibly smaller, shard."""
        if self._buffered:
            self._commit(self._buffered)

    def _check_disk(self, needed_bytes: int) -> None:
        free = shutil.disk_usage(self.directory).free
        if free - needed_bytes < self.min_free_gb * _BYTES_PER_GB:
            raise OSError(
                f"Stopping before filling the disk: {free / _BYTES_PER_GB:.1f} GB free, shard "
                f"needs {needed_bytes / _BYTES_PER_GB:.1f} GB, reserve is {self.min_free_gb} GB"
            )

    def _commit(self, contexts: int) -> None:
        tokens = np.concatenate(self._tokens)
        activations = np.concatenate(self._activations)
        shard_tokens, spare_tokens = tokens[:contexts], tokens[contexts:]
        shard_activations, spare_activations = activations[:contexts], activations[contexts:]
        self._check_disk(shard_tokens.nbytes + shard_activations.nbytes)

        index = len(self.manifest.shards)
        info = ShardInfo(index=index, contexts=contexts, activations_sha256="", tokens_sha256="")
        activations_path = self.directory / info.activations_name
        tokens_path = self.directory / info.tokens_name
        _atomic_save_array(activations_path, shard_activations)
        _atomic_save_array(tokens_path, shard_tokens)
        info.activations_sha256 = _sha256(activations_path)
        info.tokens_sha256 = _sha256(tokens_path)

        self.manifest.shards.append(info)
        temporary = self.directory / (MANIFEST_NAME + ".tmp")
        temporary.write_text(self.manifest.model_dump_json(indent=2))
        _sync_and_replace(temporary, self.directory / MANIFEST_NAME)

        self._tokens, self._activations = [spare_tokens], [spare_activations]
        self._buffered = len(spare_tokens)
        logger.info(
            "committed shard",
            extra={"fields": {"shard": index, "contexts_total": self.contexts_committed}},
        )


class ShardDataset:
    """Read-only, memory-mapped view of a cache directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = read_manifest(directory)

    def activations(self, shard: ShardInfo) -> NDArray[Any]:
        """Memory-map one shard's ``[contexts, context_size, d_model]`` activations."""
        mapped: NDArray[Any] = np.load(self.directory / shard.activations_name, mmap_mode="r")
        return mapped

    def tokens(self, shard: ShardInfo) -> NDArray[np.uint16]:
        """Memory-map one shard's ``[contexts, context_size]`` tokens."""
        mapped: NDArray[np.uint16] = np.load(self.directory / shard.tokens_name, mmap_mode="r")
        return mapped

    def verify(self) -> None:
        """Raise ``ValueError`` if any file doesn't match its recorded checksum."""
        for shard in self.manifest.shards:
            checks = (
                (shard.activations_name, shard.activations_sha256),
                (shard.tokens_name, shard.tokens_sha256),
            )
            for name, expected in checks:
                if _sha256(self.directory / name) != expected:
                    raise ValueError(f"Checksum mismatch in {self.directory / name}")
