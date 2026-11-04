"""Data-dependent initialization: input scale and decoder bias."""

from __future__ import annotations

import torch

from feature_atlas.sae.model import SparseAutoencoder


def estimate_input_scale(x: torch.Tensor) -> float:
    """Return ``s`` such that ``s·x`` has mean squared norm ``d_model``.

    Args:
        x: ``[tokens, d_model]`` raw activations.
    """
    mean_squared_norm = x.detach().cpu().double().pow(2).sum(dim=1).mean().item()
    if mean_squared_norm == 0:
        raise ValueError("Cannot scale all-zero activations")
    return float((x.shape[1] / mean_squared_norm) ** 0.5)


def geometric_median(
    points: torch.Tensor, iterations: int = 200, tol: float = 1e-7
) -> torch.Tensor:
    """Point minimizing the summed Euclidean distance to ``points``, by Weiszfeld's method.

    Each step is a mean weighted by inverse distance, so points far from the current
    estimate count less. That makes it robust to outliers in a way the mean is not.

    Runs in float64 on the CPU, since MPS has no float64 support.

    Args:
        points: ``[n, d]``.

    Returns:
        ``[d]``, float64 on the CPU.
    """
    points = points.detach().cpu().double()
    median = points.mean(dim=0)
    for _ in range(iterations):
        weights = 1.0 / (points - median).norm(dim=1).clamp_min(1e-12)
        updated = (weights[:, None] * points).sum(dim=0) / weights.sum()
        moved = (updated - median).norm()
        median = updated
        if moved <= tol * median.norm().clamp_min(1.0):
            break
    return median


@torch.no_grad()
def initialize_from_data(sae: SparseAutoencoder, sample: torch.Tensor) -> None:
    """Set the input scale, then put ``b_dec`` at the sample's geometric median.

    Subtracting ``b_dec`` before encoding centres the inputs, so features describe
    variation around a typical activation instead of each spending effort on the offset.

    Args:
        sae: The autoencoder to initialize in place.
        sample: ``[tokens, d_model]`` raw activations.
    """
    scale = estimate_input_scale(sample)
    sae.input_scale.fill_(scale)
    median = geometric_median(sample * scale)
    sae.b_dec.copy_(median.to(sae.b_dec.dtype))
