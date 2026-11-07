import torch

from feature_atlas.sae.model import SparseAutoencoder


def make_sae(d_model: int = 8, d_sae: int = 16) -> SparseAutoencoder:
    return SparseAutoencoder(d_model, d_sae)


def test_shapes() -> None:
    out = make_sae()(torch.randn(5, 8))
    assert out.reconstruction.shape == (5, 8)
    assert out.features.shape == (5, 16)
    assert out.squared_error.shape == (5,)
    assert out.l1.shape == (5,)


def test_features_are_non_negative_and_l1_matches() -> None:
    out = make_sae()(torch.randn(50, 8))
    assert (out.features >= 0).all()
    assert torch.allclose(out.l1, out.features.abs().sum(dim=1))


def test_forward_agrees_with_encode_and_decode() -> None:
    sae = make_sae()
    sae.input_scale.fill_(3.0)
    x = torch.randn(4, 8)
    out = sae(x)
    assert torch.allclose(out.features, sae.encode(x))
    assert torch.allclose(out.reconstruction, sae.decode(sae.encode(x)), atol=1e-6)


def test_squared_error_is_measured_in_scaled_units() -> None:
    sae = make_sae()
    x = torch.randn(4, 8)
    base = sae(x)
    sae.input_scale.fill_(2.0)
    # Scaling inputs and scale together leaves the scaled problem unchanged.
    scaled = sae(x / 2.0)
    assert torch.allclose(base.squared_error, scaled.squared_error, atol=1e-5)
    assert torch.allclose(base.reconstruction / 2.0, scaled.reconstruction, atol=1e-6)


def test_gradients_reach_every_parameter() -> None:
    sae = make_sae()
    out = sae(torch.randn(64, 8))
    (out.squared_error.mean() + out.l1.mean()).backward()
    for name, parameter in sae.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name


def test_decoder_starts_unit_norm_and_tied() -> None:
    sae = make_sae()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(16))
    assert torch.equal(sae.W_enc, sae.W_dec.T)


def test_projection_removes_the_radial_component() -> None:
    sae = make_sae()
    out = sae(torch.randn(64, 8))
    (out.squared_error.mean() + out.l1.mean()).backward()
    assert sae.W_dec.grad is not None
    assert (sae.W_dec.grad * sae.W_dec).sum(dim=1).abs().max() > 1e-4
    sae.project_decoder_grad()
    assert (sae.W_dec.grad * sae.W_dec).sum(dim=1).abs().max() < 1e-6


def test_unit_norm_holds_after_an_optimizer_step() -> None:
    sae = make_sae()
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.1)
    out = sae(torch.randn(64, 8))
    (out.squared_error.mean() + out.l1.mean()).backward()
    sae.project_decoder_grad()
    optimizer.step()
    assert not torch.allclose(sae.W_dec.norm(dim=1), torch.ones(16), atol=1e-4)
    sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(16), atol=1e-6)
