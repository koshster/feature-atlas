"""The SAE training loop, with checkpoint and resume.

Everything that affects the result is either recomputed from the step number (learning
rate and λ schedules) or saved in the checkpoint (weights, Adam state, firing counters,
source position, sampling generator). On a deterministic device, resuming from a shard
or tensor source therefore reproduces an uninterrupted run exactly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import torch

from feature_atlas.activations.source import ActivationSource
from feature_atlas.config import AtlasConfig, TrainConfig
from feature_atlas.sae.checkpoint import CheckpointManager
from feature_atlas.sae.init import initialize_from_data
from feature_atlas.sae.model import SparseAutoencoder
from feature_atlas.sae.resample import resample_dead_features
from feature_atlas.tracking import Tracker

logger = logging.getLogger(__name__)


def learning_rate(step: int, config: TrainConfig) -> float:
    """Linear warm-up, constant, then linear decay to zero over the final fraction."""
    warmup = min(1.0, (step + 1) / config.lr_warmup_steps) if config.lr_warmup_steps else 1.0
    total = config.total_steps
    decay_start = total * (1 - config.lr_decay_fraction)
    decay = 1.0 if step < decay_start else max(0.0, (total - step) / (total - decay_start))
    return config.lr * warmup * decay


def l1_coefficient(step: int, config: TrainConfig) -> float:
    """λ rises linearly from zero, so sparsity can't kill features before they learn."""
    warmup_steps = config.l1_warmup_fraction * config.total_steps
    if warmup_steps < 1:
        return config.l1_coefficient
    return config.l1_coefficient * min(1.0, (step + 1) / warmup_steps)


def _take(batches: Iterator[torch.Tensor], tokens: int, device: torch.device) -> torch.Tensor:
    """Concatenate batches until at least ``tokens`` rows have been collected."""
    collected: list[torch.Tensor] = []
    count = 0
    while count < tokens:
        batch = next(batches, None)
        if batch is None:
            raise RuntimeError(f"Activation source ran out after {count} of {tokens} tokens")
        collected.append(batch.to(device))
        count += len(batch)
    return torch.cat(collected)


class _Due:
    """Decides when to checkpoint: every N steps, every M minutes, or both."""

    def __init__(self, every_steps: int | None, every_minutes: float, clock: Callable[[], float]):
        self.every_steps = every_steps
        self.every_seconds = every_minutes * 60
        self.clock = clock
        self.last = clock()

    def __call__(self, completed_steps: int) -> bool:
        by_steps = self.every_steps is not None and completed_steps % self.every_steps == 0
        if by_steps or self.clock() - self.last >= self.every_seconds:
            self.last = self.clock()
            return True
        return False


def train_sae(
    config: AtlasConfig,
    source: ActivationSource,
    run_dir: Path,
    tracker: Tracker,
    device: torch.device,
    clock: Callable[[], float] = time.monotonic,
) -> SparseAutoencoder:
    """Train an SAE, resuming from the newest checkpoint in ``run_dir`` if there is one.

    Args:
        config: Full experiment config; ``train``, ``sae`` and ``run`` are used here.
        source: Activations. Its position is checkpointed along with the model.
        run_dir: Holds ``checkpoints/``. Reusing it resumes the run.
        tracker: Receives metrics every ``run.log_every_steps`` steps.
        device: Where the SAE trains.
        clock: Injected so tests can control time-based checkpointing.

    Returns:
        The trained SAE, on ``device``.
    """
    settings = config.train
    total_steps = settings.total_steps
    if total_steps < 1:
        raise ValueError("train.total_tokens must cover at least one batch")

    sae = SparseAutoencoder(source.d_model, config.sae.d_sae).to(device)
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=settings.lr, betas=(settings.adam_beta1, settings.adam_beta2)
    )
    # A dedicated CPU generator, so resampling is reproducible and its state checkpointable.
    generator = torch.Generator().manual_seed(config.seed)
    steps_since_fired = torch.zeros(config.sae.d_sae, dtype=torch.long, device=device)
    manager = CheckpointManager(
        run_dir / "checkpoints", config.training_hash(), config.run.keep_checkpoints
    )

    start_step = 0
    latest = manager.latest()
    if latest is not None:
        state = manager.load(latest)
        sae.load_state_dict(state["sae"])
        optimizer.load_state_dict(state["optimizer"])
        source.load_state_dict(state["source"])
        generator.set_state(state["generator"])
        steps_since_fired.copy_(state["steps_since_fired"])
        start_step = state["step"]
        logger.info("resumed", extra={"fields": {"step": start_step, "path": str(latest)}})

    batches = source.batches(settings.batch_size)
    if latest is None:
        sae.initialize(generator)
        initialize_from_data(sae, _take(batches, settings.init_tokens, device))

    def save(completed_steps: int) -> None:
        manager.save(
            completed_steps,
            {
                "sae": sae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "source": source.state_dict(),
                "generator": generator.get_state(),
                "steps_since_fired": steps_since_fired.cpu(),
            },
        )

    checkpoint_due = _Due(
        config.run.checkpoint_every_steps, config.run.checkpoint_every_minutes, clock
    )
    resample_until = settings.resample_until_fraction * total_steps
    interval_fires = torch.zeros(config.sae.d_sae, device=device)
    interval_tokens = 0

    for step in range(start_step, total_steps):
        x = next(batches, None)
        if x is None:
            raise RuntimeError(f"Activation source ran out at step {step} of {total_steps}")
        x = x.to(device)  # [batch, d_model]

        lr = learning_rate(step, settings)
        l1 = l1_coefficient(step, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr

        out = sae(x)
        mse = out.squared_error.mean()
        sparsity = out.l1.mean()
        loss = mse + l1 * sparsity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sae.project_decoder_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), settings.max_grad_norm)
        optimizer.step()
        sae.normalize_decoder()

        with torch.no_grad():
            fired = out.features > 0  # [batch, d_sae]
            fired_any = fired.any(dim=0)
            steps_since_fired += 1
            steps_since_fired[fired_any] = 0
            interval_fires += fired.sum(dim=0)
            interval_tokens += len(x)

        completed = step + 1
        resampled = 0
        if (
            completed % settings.resample_every_steps == 0
            and completed <= resample_until
            and completed >= settings.dead_window_steps
        ):
            dead = steps_since_fired >= settings.dead_window_steps
            inputs = _take(batches, settings.resample_tokens, device)
            resampled = resample_dead_features(sae, optimizer, dead, inputs, generator)
            # A revived feature gets a full window to prove itself before it can count as dead.
            steps_since_fired[dead] = 0

        if completed % config.run.log_every_steps == 0 or completed == total_steps or resampled:
            with torch.no_grad():
                scaled = x * sae.input_scale
                variance = (scaled - scaled.mean(dim=0)).pow(2).sum()
                frequency = interval_fires / interval_tokens
                tracker.log(
                    completed,
                    {
                        "tokens": completed * settings.batch_size,
                        "loss": loss.item(),
                        "mse": mse.item(),
                        "l1": sparsity.item(),
                        "l0": fired.sum(dim=1).float().mean().item(),
                        "explained_variance": 1 - (out.squared_error.sum() / variance).item(),
                        "dead_fraction": (steps_since_fired >= settings.dead_window_steps)
                        .float()
                        .mean()
                        .item(),
                        "dense_fraction": (frequency > settings.dense_threshold)
                        .float()
                        .mean()
                        .item(),
                        "lr": lr,
                        "l1_coefficient": l1,
                        "resampled": float(resampled),
                    },
                )
            interval_fires.zero_()
            interval_tokens = 0

        if completed == total_steps or checkpoint_due(completed):
            save(completed)

    return sae
