import torch

from feature_atlas.sae.model import SparseAutoencoder
from feature_atlas.sae.resample import resample_dead_features


def make_sae(d_model: int = 8, d_sae: int = 16) -> SparseAutoencoder:
    return SparseAutoencoder(d_model, d_sae)


def adam_with_state(sae: SparseAutoencoder) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    out = sae(torch.randn(32, 8))
    (out.squared_error.mean() + out.l1.mean()).backward()
    optimizer.step()
    return optimizer


def test_resampling_changes_only_dead_features() -> None:
    sae = make_sae()
    optimizer = adam_with_state(sae)
    before = {name: p.detach().clone() for name, p in sae.named_parameters()}
    dead = torch.zeros(16, dtype=torch.bool)
    dead[[2, 7]] = True
    count = resample_dead_features(
        sae, optimizer, dead, torch.randn(100, 8), torch.Generator().manual_seed(0)
    )
    assert count == 2
    alive = ~dead
    assert torch.equal(sae.W_dec[alive], before["W_dec"][alive])
    assert torch.equal(sae.W_enc[:, alive], before["W_enc"][:, alive])
    assert not torch.equal(sae.W_dec[dead], before["W_dec"][dead])
    assert torch.allclose(sae.W_dec[dead].norm(dim=1), torch.ones(2))
    assert torch.equal(sae.b_enc[dead], torch.zeros(2))


def test_resampling_resets_adam_moments_for_dead_features() -> None:
    sae = make_sae()
    optimizer = adam_with_state(sae)
    dead = torch.zeros(16, dtype=torch.bool)
    dead[3] = True
    resample_dead_features(sae, optimizer, dead, torch.randn(100, 8), torch.Generator())
    state = optimizer.state[sae.W_dec]
    assert torch.equal(state["exp_avg"][3], torch.zeros(8))
    assert state["exp_avg"][4].abs().sum() > 0


def test_resampled_feature_points_at_a_badly_reconstructed_input() -> None:
    sae = make_sae()
    optimizer = adam_with_state(sae)
    inputs = torch.zeros(50, 8)
    inputs[17] = torch.randn(8) * 100  # the only input with real loss
    dead = torch.zeros(16, dtype=torch.bool)
    dead[0] = True
    resample_dead_features(sae, optimizer, dead, inputs, torch.Generator().manual_seed(0))
    expected = inputs[17] - sae.b_dec
    cosine = torch.nn.functional.cosine_similarity(sae.W_dec[0], expected, dim=0)
    assert cosine > 0.999


def test_no_dead_features_changes_nothing() -> None:
    sae = make_sae()
    optimizer = adam_with_state(sae)
    before = sae.W_dec.detach().clone()
    dead = torch.zeros(16, dtype=torch.bool)
    count = resample_dead_features(sae, optimizer, dead, torch.randn(10, 8), torch.Generator())
    assert count == 0
    assert torch.equal(sae.W_dec, before)
