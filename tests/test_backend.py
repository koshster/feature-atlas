import pytest
import torch

from feature_atlas.models.backend import ModelBackend, tiny_backend


def test_activations_equal_the_hook_value_they_stand_in_for() -> None:
    backend = tiny_backend(hook_layer=1)
    tokens = torch.randint(0, 100, (3, 16))
    _, cache = backend.model.run_with_cache(tokens, names_filter="blocks.1.hook_resid_pre")
    assert torch.equal(backend.activations(tokens), cache["blocks.1.hook_resid_pre"])


def test_activations_at_layer_zero_work() -> None:
    backend = tiny_backend(hook_layer=0)
    assert backend.activations(torch.zeros(2, 16, dtype=torch.long)).shape == (2, 16, 32)


def test_later_blocks_are_not_run() -> None:
    backend = tiny_backend(hook_layer=1)
    calls: list[str] = []
    backend.model.blocks[1].register_forward_hook(lambda *_: calls.append("ran"))
    backend.activations(torch.zeros(1, 16, dtype=torch.long))
    assert calls == []


def test_activations_carry_no_gradient() -> None:
    backend = tiny_backend()
    assert not backend.activations(torch.zeros(1, 16, dtype=torch.long)).requires_grad


def test_hook_layer_past_the_end_is_rejected() -> None:
    model = tiny_backend().model
    with pytest.raises(ValueError, match="hook_layer"):
        ModelBackend(model, hook_layer=2)


def test_encode_without_tokenizer_explains_the_problem() -> None:
    with pytest.raises(ValueError, match="no tokenizer"):
        tiny_backend().encode("hello")
