from pathlib import Path

import pytest
import torch

from feature_atlas.sae.io import load_sae, save_sae
from feature_atlas.sae.model import SparseAutoencoder


def test_round_trip_keeps_weights_scale_and_metadata(tmp_path: Path) -> None:
    sae = SparseAutoencoder(8, 16)
    sae.input_scale.fill_(0.25)
    save_sae(sae, tmp_path, {"training_hash": "abc"})
    loaded, config = load_sae(tmp_path)
    assert config["training_hash"] == "abc"
    for name, tensor in sae.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name]), name


def test_missing_sae_gives_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No SAE"):
        load_sae(tmp_path)
