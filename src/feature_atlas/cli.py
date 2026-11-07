"""The ``atlas`` command line. Notebooks call this, so no logic lives only in them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
import yaml

from feature_atlas.config import load_config
from feature_atlas.runtime import (
    artifacts_root,
    configure_logging,
    seed_everything,
    select_device,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOption = Annotated[Path, typer.Option(help="Path to a YAML experiment config.")]
DeviceOption = Annotated[str, typer.Option(help="auto, cpu, mps or cuda.")]


@app.command()
def cache(
    config: ConfigOption,
    out: Annotated[Path | None, typer.Option(help="Output directory.")] = None,
    device: DeviceOption = "auto",
) -> None:
    """Cache activations to disk in shards, resuming an interrupted run."""
    from feature_atlas.activations.cache import build_cache
    from feature_atlas.models.backend import load_backend
    from feature_atlas.pipeline import text_contexts

    configure_logging()
    settings = load_config(config)
    seed_everything(settings.seed)
    backend = load_backend(settings.model, select_device(device))
    directory = out or artifacts_root() / "activations" / settings.activation_hash()
    manifest = build_cache(settings, backend, text_contexts(settings, backend, "train"), directory)
    typer.echo(f"{manifest.total_contexts} contexts in {directory}")


@app.command()
def train(config: ConfigOption, device: DeviceOption = "auto") -> None:
    """Train an SAE, resuming from the run's newest checkpoint if one exists."""
    from feature_atlas.pipeline import build_source, run_directory
    from feature_atlas.sae.io import save_sae
    from feature_atlas.sae.train import train_sae
    from feature_atlas.tracking import JsonlTracker

    configure_logging()
    settings = load_config(config)
    seed_everything(settings.seed)
    selected = select_device(device)
    run_dir = run_directory(settings)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(settings.model_dump(mode="json")))
    tracker = JsonlTracker(run_dir / "metrics.jsonl")
    try:
        sae = train_sae(settings, build_source(settings, selected), run_dir, tracker, selected)
    finally:
        tracker.close()
    save_sae(
        sae,
        run_dir / "sae",
        {
            "model": settings.model.name,
            "hook_point": settings.model.hook_point,
            "exclude_first_position": settings.source.exclude_first_position,
            "training_hash": settings.training_hash(),
        },
    )
    typer.echo(f"Saved SAE to {run_dir / 'sae'}")


@app.command(name="eval")
def evaluate(
    config: ConfigOption,
    sae_dir: Annotated[Path | None, typer.Option(help="Defaults to the run's sae/.")] = None,
    device: DeviceOption = "auto",
) -> None:
    """Evaluate an SAE on held-out text and print the metrics as JSON."""
    from feature_atlas.eval.evaluate import evaluate_sae
    from feature_atlas.models.backend import load_backend
    from feature_atlas.pipeline import heldout_batches, run_directory
    from feature_atlas.sae.io import load_sae

    configure_logging()
    settings = load_config(config)
    seed_everything(settings.seed)
    selected = select_device(device)
    directory = sae_dir or run_directory(settings) / "sae"
    sae, _ = load_sae(directory)
    backend = load_backend(settings.model, selected)
    result = evaluate_sae(
        sae.to(selected),
        backend,
        heldout_batches(settings, backend),
        exclude_first_position=settings.source.exclude_first_position,
        dense_threshold=settings.train.dense_threshold,
    )
    (directory / "eval.json").write_text(result.model_dump_json(indent=2))
    typer.echo(result.model_dump_json())


@app.command()
def benchmark(
    config: ConfigOption,
    device: DeviceOption = "auto",
    batch_size: Annotated[int, typer.Option(help="Contexts per batch.")] = 32,
    batches: Annotated[int, typer.Option(help="Batches to time.")] = 20,
) -> None:
    """Report activation throughput and free disk, as JSON."""
    from feature_atlas.activations.benchmark import measure_throughput
    from feature_atlas.models.backend import load_backend

    settings = load_config(config)
    seed_everything(settings.seed)
    selected = select_device(device)
    backend = load_backend(settings.model, selected)
    speed = measure_throughput(backend, batch_size, settings.data.context_size, batches)
    result = {
        "device": str(selected),
        "batch_size": batch_size,
        "tokens_per_second": round(speed),
        "free_disk_gb": round(shutil.disk_usage(artifacts_root()).free / 1024**3, 1),
    }
    typer.echo(json.dumps(result))
