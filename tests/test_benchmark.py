from feature_atlas.activations.benchmark import measure_throughput
from feature_atlas.models.backend import tiny_backend


def test_throughput_is_positive() -> None:
    speed = measure_throughput(tiny_backend(), batch_size=2, context_size=16, batches=2)
    assert speed > 0
