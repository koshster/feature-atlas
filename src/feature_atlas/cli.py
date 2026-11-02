"""The ``atlas`` command line. Notebooks call this, so no logic lives only in them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer

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
    from feature_atlas.data.text import iter_contexts, load_documents
    from feature_atlas.models.backend import load_backend

    configure_logging()
    settings = load_config(config)
    seed_everything(settings.seed)
    backend = load_backend(settings.model, select_device(device))
    contexts = iter_contexts(
        load_documents(settings.data, "train"),
        backend.encode,
        settings.data.context_size,
        backend.eos_token_id,
        backend.bos_token_id if settings.data.prepend_bos else None,
    )
    directory = out or artifacts_root() / "activations" / settings.activation_hash()
    manifest = build_cache(settings, backend, contexts, directory)
    typer.echo(f"{manifest.total_contexts} contexts in {directory}")


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
