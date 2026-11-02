from pathlib import Path

import pytest

from feature_atlas.config import AtlasConfig, CacheConfig, load_config, stable_hash


def test_hook_point_follows_layer() -> None:
    assert AtlasConfig().model.hook_point == "blocks.8.hook_resid_pre"


def test_load_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 3\nmodel:\n  hook_layer: 2\n")
    config = load_config(path)
    assert config.seed == 3
    assert config.model.hook_layer == 2


def test_unknown_key_is_rejected_and_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("modle:\n  hook_layer: 2\n")
    with pytest.raises(ValueError, match=r"bad\.yaml"):
        load_config(path)


def test_missing_file_gives_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot read config"):
        load_config(tmp_path / "missing.yaml")


def test_shard_larger_than_total_is_rejected() -> None:
    with pytest.raises(ValueError, match="shard_contexts"):
        CacheConfig(total_contexts=10, shard_contexts=20)


def test_stable_hash_ignores_key_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_activation_hash_tracks_values_but_not_total_size() -> None:
    base = AtlasConfig()
    extended = base.model_copy(
        update={"cache": base.cache.model_copy(update={"total_contexts": 9_999})}
    )
    changed = base.model_copy(
        update={"model": base.model.model_copy(update={"center_writing_weights": False})}
    )
    assert base.activation_hash() == extended.activation_hash()
    assert base.activation_hash() != changed.activation_hash()
