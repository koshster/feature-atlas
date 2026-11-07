"""Revive dead features, following Anthropic's *Towards Monosemanticity* (2023).

A ReLU feature that never fires gets zero gradient, so it can never recover by itself.
Resampling points it at an input the SAE currently reconstructs badly, which gives it
an immediate job.
"""

from __future__ import annotations

import torch

from feature_atlas.sae.model import SparseAutoencoder

# Encoder rows are set to this fraction of the alive features' mean encoder norm, so a
# revived feature starts weak and can't immediately disrupt features that already work.
ENCODER_NORM_FRACTION = 0.2


@torch.no_grad()
def resample_dead_features(
    sae: SparseAutoencoder,
    optimizer: torch.optim.Adam,
    dead: torch.Tensor,
    inputs: torch.Tensor,
    generator: torch.Generator,
) -> int:
    """Re-point dead features at badly reconstructed inputs, in place.

    Inputs are drawn with probability proportional to squared reconstruction loss, so
    the worst-explained inputs are most likely to get a new feature.

    Args:
        sae: The autoencoder to modify.
        optimizer: Its Adam optimizer. Moments for the changed weights are reset, since
            momentum from the old, useless direction would immediately pull it back.
        dead: ``[d_sae]`` boolean mask.
        inputs: ``[tokens, d_model]`` raw activations to sample from.
        generator: CPU generator, so sampling is reproducible and checkpointable.

    Returns:
        The number of features resampled.
    """
    dead_indices = dead.nonzero().squeeze(1)
    if len(dead_indices) == 0:
        return 0
    losses = sae(inputs).squared_error
    weights = losses.pow(2).cpu().double()
    if weights.sum() == 0:
        return 0
    chosen = torch.multinomial(
        weights, len(dead_indices), replacement=len(dead_indices) > len(inputs), generator=generator
    ).to(inputs.device)

    # The encoder sees inputs relative to b_dec, so directions are taken relative to it too.
    centred = inputs[chosen] * sae.input_scale - sae.b_dec  # [n_dead, d_model]
    directions = centred / centred.norm(dim=1, keepdim=True).clamp_min(1e-8)
    alive = ~dead
    encoder_norm = sae.W_enc[:, alive].norm(dim=0).mean() if alive.any() else torch.tensor(1.0)

    sae.W_dec[dead_indices] = directions
    sae.W_enc[:, dead_indices] = directions.T * encoder_norm * ENCODER_NORM_FRACTION
    sae.b_enc[dead_indices] = 0.0

    for parameter, index in (
        (sae.W_dec, (dead_indices,)),
        (sae.W_enc, (slice(None), dead_indices)),
        (sae.b_enc, (dead_indices,)),
    ):
        state = optimizer.state.get(parameter, {})
        for moment in ("exp_avg", "exp_avg_sq"):
            if moment in state:
                state[moment][index] = 0.0
    return len(dead_indices)
