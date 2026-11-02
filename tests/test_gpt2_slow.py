"""Checks against real GPT-2. These download weights, so they carry the ``slow`` marker."""

import pytest
import torch

from feature_atlas.config import ModelConfig
from feature_atlas.models.backend import load_backend

pytestmark = pytest.mark.slow

TEXT = "The Eiffel Tower is located in the city of"


@pytest.fixture(scope="module")
def backend():  # type: ignore[no-untyped-def]
    return load_backend(ModelConfig(), torch.device("cpu"))


def test_processed_model_predicts_like_hugging_face(backend) -> None:  # type: ignore[no-untyped-def]
    # Centring shifts logits by a constant, so compare log-probabilities, not logits.
    from transformers import GPT2LMHeadModel

    reference = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tokens = torch.tensor([backend.encode(TEXT)])
    with torch.inference_mode():
        ours = backend.model(tokens).log_softmax(-1)
        theirs = reference(tokens).logits.log_softmax(-1)
    assert torch.allclose(ours, theirs, atol=1e-3)


def test_real_activation_shape(backend) -> None:  # type: ignore[no-untyped-def]
    tokens = torch.tensor([backend.encode(TEXT)])
    assert backend.activations(tokens).shape == (1, tokens.shape[1], 768)


def test_float16_rounding_is_small(backend) -> None:  # type: ignore[no-untyped-def]
    # Rounding must sit far below SAE reconstruction error, or the cache would limit it.
    tokens = torch.tensor([backend.encode(TEXT * 5)])
    activations = backend.activations(tokens)
    rounded = activations.half().float()
    relative = (rounded - activations).norm() / activations.norm()
    assert torch.isfinite(rounded).all()
    assert relative < 1e-3
