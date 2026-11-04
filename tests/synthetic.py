"""Synthetic data with a known sparse dictionary, for testing SAE training."""

import torch


def sparse_dictionary_data(
    tokens: int = 50_000,
    d_model: int = 32,
    n_features: int = 48,
    active_probability: float = 0.06,
    seed: int = 12_345,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[tokens, d_model]`` data built from ``[n_features, d_model]`` unit directions.

    Each token is a non-negative sparse combination of the directions: about
    ``n_features * active_probability`` are active per token. An SAE that works should
    find directions matching the planted ones.

    The default seed differs from the SAE's initialization seed on purpose. With both at
    0, the SAE's random initial directions *were* the planted ones, and an untrained SAE
    "recovered" them perfectly.
    """
    generator = torch.Generator().manual_seed(seed)
    dictionary = torch.randn(n_features, d_model, generator=generator)
    dictionary /= dictionary.norm(dim=1, keepdim=True)
    active = torch.rand(tokens, n_features, generator=generator) < active_probability
    magnitudes = torch.rand(tokens, n_features, generator=generator) + 0.5
    return (active * magnitudes) @ dictionary, dictionary


def mean_max_cosine(found: torch.Tensor, planted: torch.Tensor) -> torch.Tensor:
    """For each planted direction, the best cosine similarity among found directions."""
    found = found / found.norm(dim=1, keepdim=True)
    return (planted @ found.T).max(dim=1).values
