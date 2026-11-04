"""A ReLU sparse autoencoder with a unit-norm dictionary.

    f(x) = ReLU((s·x - b_dec) W_enc + b_enc)      # features, [batch, d_sae]
    x̂    = (f W_dec + b_dec) / s                  # reconstruction, [batch, d_model]
    L    = mean ||s·x - s·x̂||² + λ mean ||f||₁

``s`` is a fixed input scale chosen so that scaled inputs have mean squared norm
``d_model``. It makes λ and the learning rate independent of the model's activation
scale, so they transfer between layers and models. The loss is computed in scaled
units, and callers only ever see raw-scale inputs and reconstructions.

Each row of ``W_dec`` is one feature's direction (a column in the usual matrix
notation). Rows are kept at unit norm: otherwise the L1 penalty could be dodged by
shrinking ``f`` and growing ``W_dec`` to compensate, making sparsity an illusion.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class SAEOutput(NamedTuple):
    reconstruction: torch.Tensor  # [batch, d_model], raw scale
    features: torch.Tensor  # [batch, d_sae], in scaled units
    squared_error: torch.Tensor  # [batch], ||s·x - s·x̂||²
    l1: torch.Tensor  # [batch], ||f||₁


class SparseAutoencoder(nn.Module):
    """Encoder, decoder and the per-token terms of the training loss."""

    def __init__(self, d_model: int, d_sae: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # A buffer, not a parameter: it is set once from data, then saved with the weights.
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.initialize(torch.Generator().manual_seed(0))

    input_scale: torch.Tensor

    @torch.no_grad()
    def initialize(self, generator: torch.Generator) -> None:
        """Random unit-norm dictionary, with the encoder initialized to its transpose.

        Starting tied means each feature initially responds to its own direction, which
        speeds early training and reduces features that die before learning anything.
        """
        directions = torch.randn(self.d_sae, self.d_model, generator=generator)
        self.W_dec.copy_(directions / directions.norm(dim=1, keepdim=True))
        self.W_enc.copy_(self.W_dec.T)
        self.b_enc.zero_()
        self.b_dec.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw ``[batch, d_model]`` activations to ``[batch, d_sae]`` features."""
        return torch.relu((x * self.input_scale - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, d_sae]`` features to raw-scale ``[batch, d_model]`` activations."""
        return (features @ self.W_dec + self.b_dec) / self.input_scale

    def forward(self, x: torch.Tensor) -> SAEOutput:
        scaled = x * self.input_scale
        features = torch.relu((scaled - self.b_dec) @ self.W_enc + self.b_enc)
        reconstruction = features @ self.W_dec + self.b_dec
        return SAEOutput(
            reconstruction=reconstruction / self.input_scale,
            features=features,
            squared_error=(scaled - reconstruction).pow(2).sum(dim=-1),
            l1=features.sum(dim=-1),  # features are non-negative, so this is the L1 norm
        )

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Rescale each dictionary row back to unit norm, after every optimizer step."""
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        """Remove each row gradient's component along the row, before the optimizer step.

        For a unit row ``w``, that component only changes ``||w||``, which
        ``normalize_decoder`` undoes anyway. Left in, it would still enter Adam's moment
        estimates and distort the size of the useful, tangential updates.
        """
        grad = self.W_dec.grad
        if grad is not None:
            # Assumes unit rows, which normalize_decoder guarantees after every step.
            grad -= (grad * self.W_dec).sum(dim=1, keepdim=True) * self.W_dec
