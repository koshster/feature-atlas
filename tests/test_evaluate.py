import torch

from feature_atlas.eval.evaluate import evaluate_sae
from feature_atlas.models.backend import tiny_backend
from feature_atlas.sae.model import SparseAutoencoder


def identity_sae(d_model: int) -> SparseAutoencoder:
    """An SAE that reconstructs perfectly, using relu(x) - relu(-x) = x."""
    sae = SparseAutoencoder(d_model, 2 * d_model)
    eye = torch.eye(d_model)
    with torch.no_grad():
        sae.W_enc.copy_(torch.cat([eye, -eye], dim=1))
        sae.W_dec.copy_(torch.cat([eye, -eye], dim=0))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def batches() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [torch.randint(0, 100, (4, 16), generator=generator) for _ in range(3)]


def test_patching_nothing_matches_the_clean_loss() -> None:
    backend = tiny_backend()
    tokens = batches()[0]
    clean = backend.next_token_loss(tokens)
    assert clean.shape == (4, 15)
    assert torch.allclose(backend.next_token_loss(tokens, lambda a: a), clean)
    assert not torch.allclose(backend.next_token_loss(tokens, torch.zeros_like), clean)


def test_perfect_sae_recovers_all_loss() -> None:
    result = evaluate_sae(identity_sae(32), tiny_backend(), batches())
    assert result.tokens == 3 * 4 * 15
    assert result.explained_variance > 0.9999
    assert abs(result.loss_recovered - 1.0) < 1e-4
    assert result.ce_zero != result.ce_clean


def test_untrained_sae_recovers_less_than_a_perfect_one() -> None:
    result = evaluate_sae(SparseAutoencoder(32, 64), tiny_backend(), batches())
    assert result.loss_recovered < 0.999
    assert 0 < result.l0 <= 64
