"""Measure how fast the model produces activations on this machine."""

from __future__ import annotations

import time

import torch

from feature_atlas.models.backend import ModelBackend


def _wait_for_device(device: torch.device) -> None:
    # GPU work is queued asynchronously, so without this the timer would stop early.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_throughput(
    backend: ModelBackend,
    batch_size: int,
    context_size: int,
    batches: int,
    warmup_batches: int = 2,
) -> float:
    """Return activation tokens per second on random tokens.

    Random tokens measure model speed alone, excluding tokenization and disk. The first
    batches are skipped because they include one-off costs such as allocator setup.
    """
    d_vocab = int(backend.model.cfg.d_vocab)
    tokens = torch.randint(0, d_vocab, (batch_size, context_size), device=backend.device)
    for _ in range(warmup_batches):
        backend.activations(tokens)
    _wait_for_device(backend.device)
    start = time.perf_counter()
    for _ in range(batches):
        backend.activations(tokens)
    _wait_for_device(backend.device)
    return batches * tokens.numel() / (time.perf_counter() - start)
