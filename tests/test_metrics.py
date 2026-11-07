import pytest
import torch

from feature_atlas.eval.metrics import FeatureStats, ReconstructionStats, loss_recovered


def test_perfect_reconstruction_explains_everything() -> None:
    stats = ReconstructionStats(4)
    x = torch.randn(100, 4)
    stats.update(x, x)
    assert stats.explained_variance == pytest.approx(1.0)
    assert stats.mse == 0


def test_predicting_the_mean_explains_nothing() -> None:
    stats = ReconstructionStats(4)
    x = torch.randn(100, 4) + 7
    stats.update(x, x.mean(dim=0).expand_as(x))
    assert stats.explained_variance == pytest.approx(0.0, abs=1e-9)


def test_streaming_matches_one_big_batch() -> None:
    x = torch.randn(300, 4) * torch.tensor([1.0, 5.0, 0.1, 2.0]) + 3
    noisy = x + torch.randn(300, 4) * 0.3
    whole = ReconstructionStats(4)
    whole.update(x, noisy)
    streamed = ReconstructionStats(4)
    for chunk, chunk_noisy in zip(x.split(7), noisy.split(7), strict=True):
        streamed.update(chunk, chunk_noisy)
    assert streamed.explained_variance == pytest.approx(whole.explained_variance, rel=1e-12)


def test_feature_stats() -> None:
    stats = FeatureStats(3)
    stats.update(torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.5, 0.0]]))
    stats.update(torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert stats.l0 == pytest.approx(5 / 4)
    assert stats.frequency.tolist() == [1.0, 0.25, 0.0]
    assert stats.dead_fraction() == pytest.approx(1 / 3)
    assert stats.dense_fraction(0.5) == pytest.approx(1 / 3)


def test_loss_recovered_endpoints() -> None:
    assert loss_recovered(ce_clean=3.0, ce_sae=3.0, ce_zero=10.0) == 1.0
    assert loss_recovered(ce_clean=3.0, ce_sae=10.0, ce_zero=10.0) == 0.0
    assert loss_recovered(ce_clean=3.0, ce_sae=4.0, ce_zero=10.0) == pytest.approx(6 / 7)
