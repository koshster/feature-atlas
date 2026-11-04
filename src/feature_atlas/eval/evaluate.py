"""Evaluate an SAE on held-out text: reconstruction, sparsity and downstream loss."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from pydantic import BaseModel

from feature_atlas.eval.metrics import FeatureStats, ReconstructionStats, loss_recovered
from feature_atlas.models.backend import ModelBackend
from feature_atlas.sae.model import SparseAutoencoder


class EvalResult(BaseModel):
    tokens: int
    explained_variance: float
    mse: float
    l0: float
    dead_fraction: float
    dense_fraction: float
    ce_clean: float
    ce_sae: float
    ce_zero: float
    loss_recovered: float


@torch.inference_mode()
def evaluate_sae(
    sae: SparseAutoencoder,
    backend: ModelBackend,
    token_batches: Iterable[torch.Tensor],
    exclude_first_position: bool = True,
    dense_threshold: float = 0.1,
) -> EvalResult:
    """Measure an SAE on ``[batch, context]`` token batches.

    When position 0 is excluded, both the SAE splice and the zero ablation leave it
    untouched, and reconstruction metrics ignore it. The SAE never trained on it, and
    zeroing it alone would change the loss by a lot that has nothing to do with the SAE.
    Cross-entropy is still averaged over every predicted token, so the three losses
    are directly comparable.
    """
    start = 1 if exclude_first_position else 0
    reconstruction = ReconstructionStats(sae.d_model)
    features = FeatureStats(sae.d_sae)
    losses: dict[str, list[torch.Tensor]] = {"clean": [], "sae": [], "zero": []}

    def splice(activations: torch.Tensor) -> torch.Tensor:
        patched = activations.clone()
        kept = activations[:, start:]  # [batch, positions, d_model]
        out = sae(kept.flatten(0, 1))
        reconstruction.update(kept.flatten(0, 1), out.reconstruction)
        features.update(out.features)
        patched[:, start:] = out.reconstruction.view_as(kept)
        return patched

    def zero(activations: torch.Tensor) -> torch.Tensor:
        patched = activations.clone()
        patched[:, start:] = 0.0
        return patched

    for tokens in token_batches:
        losses["clean"].append(backend.next_token_loss(tokens).flatten().cpu())
        losses["sae"].append(backend.next_token_loss(tokens, splice).flatten().cpu())
        losses["zero"].append(backend.next_token_loss(tokens, zero).flatten().cpu())

    if not features.tokens:
        raise ValueError("No tokens to evaluate")
    ce = {name: torch.cat(values).double().mean().item() for name, values in losses.items()}
    return EvalResult(
        tokens=features.tokens,
        explained_variance=reconstruction.explained_variance,
        mse=reconstruction.mse,
        l0=features.l0,
        dead_fraction=features.dead_fraction(),
        dense_fraction=features.dense_fraction(dense_threshold),
        ce_clean=ce["clean"],
        ce_sae=ce["sae"],
        ce_zero=ce["zero"],
        loss_recovered=loss_recovered(ce["clean"], ce["sae"], ce["zero"]),
    )
