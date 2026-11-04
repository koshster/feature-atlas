"""Streaming accumulators for SAE quality metrics.

Metrics are accumulated over many batches in float64, because averaging per-batch
ratios such as explained variance would give a different, biased answer.
"""

from __future__ import annotations

import torch


class ReconstructionStats:
    """Explained variance and mean squared error over a stream of batches.

    Explained variance is ``1 - Σ||x - x̂||² / Σ||x - mean(x)||²``, with the mean taken over
    every token seen. The total sum of squares is recovered from running sums as
    ``Σx² - n·mean²`` per dimension, so the global mean needn't be known in advance.
    """

    def __init__(self, d_model: int) -> None:
        self.tokens = 0
        self.residual_sum_squares = 0.0
        self.sum = torch.zeros(d_model, dtype=torch.float64)
        self.sum_squares = torch.zeros(d_model, dtype=torch.float64)

    def update(self, x: torch.Tensor, reconstruction: torch.Tensor) -> None:
        """Add ``[tokens, d_model]`` inputs and reconstructions."""
        x64 = x.detach().double().cpu()
        self.tokens += len(x64)
        self.residual_sum_squares += (
            (x64 - reconstruction.detach().double().cpu()).pow(2).sum().item()
        )
        self.sum += x64.sum(dim=0)
        self.sum_squares += x64.pow(2).sum(dim=0)

    @property
    def explained_variance(self) -> float:
        total = (self.sum_squares - self.sum.pow(2) / self.tokens).sum().item()
        return 1.0 - self.residual_sum_squares / total

    @property
    def mse(self) -> float:
        """Mean squared error per token, summed over dimensions, in raw units."""
        return self.residual_sum_squares / self.tokens


class FeatureStats:
    """Firing frequency per feature, and mean L0, over a stream of batches."""

    def __init__(self, d_sae: int) -> None:
        self.tokens = 0
        self.fires = torch.zeros(d_sae, dtype=torch.float64)

    def update(self, features: torch.Tensor) -> None:
        """Add ``[tokens, d_sae]`` feature activations."""
        fired = (features.detach() > 0).double().cpu()
        self.tokens += len(fired)
        self.fires += fired.sum(dim=0)

    @property
    def frequency(self) -> torch.Tensor:
        """``[d_sae]`` fraction of tokens on which each feature fired."""
        return self.fires / self.tokens

    @property
    def l0(self) -> float:
        """Mean number of active features per token."""
        return self.fires.sum().item() / self.tokens

    def dead_fraction(self) -> float:
        """Fraction of features that never fired on this data."""
        return (self.fires == 0).double().mean().item()

    def dense_fraction(self, threshold: float) -> float:
        """Fraction of features firing on more than ``threshold`` of tokens."""
        return (self.frequency > threshold).double().mean().item()


def loss_recovered(ce_clean: float, ce_sae: float, ce_zero: float) -> float:
    """Share of the zero-ablation loss gap closed by the SAE reconstruction.

    ``(CE_zero - CE_sae) / (CE_zero - CE_clean)``: 1 means splicing in the SAE costs the
    model nothing, and 0 means it's as bad as deleting the activation entirely.
    """
    return (ce_zero - ce_sae) / (ce_zero - ce_clean)
