"""Save and load trained SAEs as safetensors plus a JSON config.

Safetensors holds only tensors, so loading a downloaded SAE can't execute code, unlike
a pickle. That matters once weights are shared on the Hugging Face Hub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from feature_atlas.sae.model import SparseAutoencoder

WEIGHTS_NAME = "sae.safetensors"
CONFIG_NAME = "sae_config.json"


def save_sae(sae: SparseAutoencoder, directory: Path, metadata: dict[str, Any]) -> None:
    """Write weights and a config describing the SAE and how it was trained."""
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: tensor.detach().cpu().contiguous() for name, tensor in sae.state_dict().items()
    }
    save_file(tensors, directory / WEIGHTS_NAME)
    config = {"d_model": sae.d_model, "d_sae": sae.d_sae, **metadata}
    (directory / CONFIG_NAME).write_text(json.dumps(config, indent=2))


def load_sae(directory: Path) -> tuple[SparseAutoencoder, dict[str, Any]]:
    """Load an SAE saved by ``save_sae``, returning it with its config."""
    config_path = directory / CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"No SAE at {directory}: {CONFIG_NAME} is missing")
    config: dict[str, Any] = json.loads(config_path.read_text())
    sae = SparseAutoencoder(config["d_model"], config["d_sae"])
    sae.load_state_dict(load_file(directory / WEIGHTS_NAME))
    return sae, config
