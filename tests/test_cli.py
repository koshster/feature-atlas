from pathlib import Path

from typer.testing import CliRunner

from feature_atlas.cli import app


def test_help_lists_commands() -> None:
    output = CliRunner().invoke(app, ["--help"]).output
    for command in ("cache", "train", "eval", "benchmark"):
        assert command in output


def test_bad_config_fails_with_the_file_named(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("nonsense: 1\n")
    result = CliRunner().invoke(app, ["cache", "--config", str(path)])
    assert result.exit_code != 0
    assert "bad.yaml" in str(result.exception)


def test_train_on_a_shard_cache_saves_an_sae(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    from feature_atlas.activations.shards import ShardWriter
    from feature_atlas.sae.io import load_sae
    from feature_atlas.tracking import read_metrics

    writer = ShardWriter(
        tmp_path / "cache", "h", d_model=8, context_size=4, dtype="float32", shard_contexts=50
    )
    writer.add(np.zeros((200, 4)), np.random.default_rng(0).normal(size=(200, 4, 8)))
    config = tmp_path / "train.yaml"
    config.write_text(
        f"""
source: {{kind: shards, shard_dir: {tmp_path / "cache"}}}
sae: {{d_sae: 16}}
train: {{total_tokens: 3000, batch_size: 100, init_tokens: 200, lr_warmup_steps: 2}}
run: {{name: cli-test, log_every_steps: 10}}
"""
    )
    monkeypatch.setenv("FEATURE_ATLAS_HOME", str(tmp_path / "home"))
    result = CliRunner().invoke(app, ["train", "--config", str(config), "--device", "cpu"])
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "home" / "runs" / "cli-test"
    sae, metadata = load_sae(run_dir / "sae")
    assert (sae.d_model, sae.d_sae) == (8, 16)
    assert metadata["exclude_first_position"] is True
    assert [entry["step"] for entry in read_metrics(run_dir / "metrics.jsonl")] == [10, 20, 30]
    assert (run_dir / "checkpoints" / "step_000000030.pt").exists()
