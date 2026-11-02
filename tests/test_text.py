import torch

from feature_atlas.data.text import assign_split, iter_contexts, iter_token_batches


def encode(text: str) -> list[int]:
    return [int(word) for word in text.split()]


def test_split_is_deterministic_and_salted() -> None:
    assert assign_split("doc", 0.5, "a") == assign_split("doc", 0.5, "a")
    results = {assign_split("doc", 0.5, str(salt)) for salt in range(50)}
    assert results == {"train", "holdout"}


def test_split_fraction_is_approximately_right() -> None:
    held = sum(assign_split(f"doc {i}", 0.1, "s") == "holdout" for i in range(10_000))
    assert 850 < held < 1150


def test_zero_fraction_holds_nothing_out() -> None:
    assert all(assign_split(f"doc {i}", 0.0, "s") == "train" for i in range(100))


def test_contexts_concatenate_documents_with_eos() -> None:
    contexts = list(iter_contexts(["1 2", "3 4 5"], encode, context_size=3, eos_token_id=9))
    assert contexts == [[1, 2, 9], [3, 4, 5]]


def test_contexts_start_with_bos_and_keep_their_size() -> None:
    contexts = list(
        iter_contexts(["1 2 3 4 5"], encode, context_size=3, eos_token_id=9, bos_token_id=8)
    )
    assert contexts == [[8, 1, 2], [8, 3, 4], [8, 5, 9]]


def test_incomplete_final_context_is_dropped() -> None:
    assert list(iter_contexts(["1"], encode, context_size=3, eos_token_id=9)) == []


def test_token_batches_keep_a_short_final_batch() -> None:
    batches = list(iter_token_batches([[1, 2], [3, 4], [5, 6]], batch_size=2))
    assert [batch.shape for batch in batches] == [torch.Size([2, 2]), torch.Size([1, 2])]
    assert batches[0].dtype == torch.long
