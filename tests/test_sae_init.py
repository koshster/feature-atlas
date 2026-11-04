import pytest
import torch

from feature_atlas.sae.init import estimate_input_scale, geometric_median, initialize_from_data
from feature_atlas.sae.model import SparseAutoencoder


def make_sae(d_model: int = 8, d_sae: int = 16) -> SparseAutoencoder:
    return SparseAutoencoder(d_model, d_sae)


def test_input_scale_gives_mean_squared_norm_d_model() -> None:
    x = torch.randn(1000, 8) * 37.0
    scale = estimate_input_scale(x)
    assert (x * scale).pow(2).sum(dim=1).mean().item() == pytest.approx(8.0, rel=1e-4)


def test_zero_inputs_cannot_be_scaled() -> None:
    with pytest.raises(ValueError, match="all-zero"):
        estimate_input_scale(torch.zeros(4, 8))


def test_geometric_median_resists_outliers() -> None:
    points = torch.cat([torch.zeros(99, 2), torch.tensor([[1000.0, 1000.0]])])
    assert geometric_median(points).norm() < 1e-3
    assert points.mean(dim=0).norm() > 10


def test_geometric_median_minimizes_total_distance() -> None:
    points = torch.randn(200, 3, generator=torch.Generator().manual_seed(0))
    median = geometric_median(points)
    best = (points.double() - median).norm(dim=1).sum()
    for nudge in torch.eye(3, dtype=torch.float64) * 0.01:
        assert best <= (points.double() - (median + nudge)).norm(dim=1).sum()
        assert best <= (points.double() - (median - nudge)).norm(dim=1).sum()


def test_initialize_from_data_sets_scale_and_bias() -> None:
    sae = make_sae()
    sample = torch.randn(500, 8) * 5 + 40
    initialize_from_data(sae, sample)
    assert sae.input_scale.item() == pytest.approx(estimate_input_scale(sample), rel=1e-6)
    expected = geometric_median(sample * sae.input_scale).float()
    assert torch.allclose(sae.b_dec, expected, atol=1e-4)
