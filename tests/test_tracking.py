from pathlib import Path

from feature_atlas.tracking import JsonlTracker, read_metrics


def test_jsonl_round_trip_appends_across_sessions(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "metrics.jsonl"
    for loss in (5.0, 4.0):
        tracker = JsonlTracker(path)
        tracker.log(1, {"loss": loss})
        tracker.close()
    assert len(path.read_text().splitlines()) == 2


def test_read_metrics_keeps_the_last_entry_per_step(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "loss": 5}\n{"step": 2, "loss": 4}\n{"step": 2, "loss": 3}\n')
    assert [entry["loss"] for entry in read_metrics(path)] == [5, 3]
