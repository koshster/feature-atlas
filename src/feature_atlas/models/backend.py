"""The only module that imports TransformerLens.

Everything else sees plain tensors, so moving off the deprecated ``HookedTransformer``
later means changing this file alone.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from feature_atlas.config import ModelConfig


@contextmanager
def _quiet_deprecation() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="HookedTransformer", category=DeprecationWarning)
        yield


class ModelBackend:
    """Wraps a hooked model and exposes activations at one residual-stream site."""

    def __init__(self, model: HookedTransformer, hook_layer: int) -> None:
        if hook_layer >= model.cfg.n_layers:
            raise ValueError(f"hook_layer {hook_layer} needs a model with more layers")
        self.model = model
        self.hook_layer = hook_layer
        self.model.eval()

    @property
    def d_model(self) -> int:
        return int(self.model.cfg.d_model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def eos_token_id(self) -> int:
        return int(self._tokenizer().eos_token_id)

    @property
    def bos_token_id(self) -> int:
        return int(self._tokenizer().bos_token_id)

    def _tokenizer(self) -> Any:
        if self.model.tokenizer is None:
            raise ValueError("This model has no tokenizer; pass token tensors instead")
        return self.model.tokenizer

    def encode(self, text: str) -> list[int]:
        """Tokenize text without adding special tokens."""
        ids: list[int] = self._tokenizer().encode(text, add_special_tokens=False)
        return ids

    @torch.inference_mode()
    def activations(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the residual stream entering block ``hook_layer``.

        ``stop_at_layer=N`` stops *before* block N, so block N's ``hook_resid_pre`` never
        fires and ``run_with_cache`` would capture nothing. The tensor it returns is
        exactly the value that hook would have seen, and skipping the later blocks saves
        about a third of the forward pass at layer 8 of 12.

        Args:
            tokens: ``[batch, context]`` integer tokens.

        Returns:
            ``[batch, context, d_model]`` activations.
        """
        output: torch.Tensor = self.model(tokens.to(self.device), stop_at_layer=self.hook_layer)
        return output

    @property
    def hook_point(self) -> str:
        return f"blocks.{self.hook_layer}.hook_resid_pre"

    @torch.inference_mode()
    def next_token_loss(
        self,
        tokens: torch.Tensor,
        patch: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the full model and return per-token cross-entropy.

        Args:
            tokens: ``[batch, context]`` integer tokens.
            patch: Optional function replacing the ``[batch, context, d_model]`` residual
                stream at ``hook_point``, for splicing in reconstructions or ablations.

        Returns:
            ``[batch, context - 1]`` losses for predicting each next token.
        """

        def hook(activations: torch.Tensor, hook: object) -> torch.Tensor:
            assert patch is not None
            return patch(activations)

        hooks: list[tuple[str | Callable[..., Any], Callable[..., Any]]] = (
            [] if patch is None else [(self.hook_point, hook)]
        )
        losses: torch.Tensor = self.model.run_with_hooks(
            tokens.to(self.device), return_type="loss", loss_per_token=True, fwd_hooks=hooks
        )
        return losses


def load_backend(config: ModelConfig, device: torch.device) -> ModelBackend:
    """Download a pretrained model with the activation settings from ``config``."""
    with _quiet_deprecation():
        model = HookedTransformer.from_pretrained(
            config.name,
            device=str(device),
            fold_ln=config.fold_ln,
            center_writing_weights=config.center_writing_weights,
            center_unembed=config.center_unembed,
            fold_value_biases=config.fold_value_biases,
        )
    return ModelBackend(model, config.hook_layer)


def tiny_backend(
    hook_layer: int = 1, d_model: int = 32, context_size: int = 16, d_vocab: int = 100
) -> ModelBackend:
    """Build a small random model so tests need no downloads."""
    config = HookedTransformerConfig(
        n_layers=2,
        d_model=d_model,
        d_head=8,
        n_heads=d_model // 8,
        n_ctx=context_size,
        d_vocab=d_vocab,
        d_mlp=d_model * 2,
        act_fn="gelu",
        seed=0,
        device="cpu",
    )
    with _quiet_deprecation():
        return ModelBackend(HookedTransformer(config), hook_layer)
