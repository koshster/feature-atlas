from pathlib import Path

from typer.testing import CliRunner

from feature_atlas.cli import app


def test_help_lists_commands() -> None:
    output = CliRunner().invoke(app, ["--help"]).output
    assert "cache" in output
    assert "benchmark" in output


def test_bad_config_fails_with_the_file_named(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("nonsense: 1\n")
    result = CliRunner().invoke(app, ["cache", "--config", str(path)])
    assert result.exit_code != 0
    assert "bad.yaml" in str(result.exception)
