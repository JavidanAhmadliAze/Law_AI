from evaluation.retrieval.metrics import (
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_perfect_retrieval() -> None:
    retrieved = ["Art. 54", "Art. 30"]
    expected = {"Art. 54", "Art. 30"}
    assert recall_at_k(retrieved, expected, 5) == 1.0
    assert mrr(retrieved, expected) == 1.0
    assert ndcg_at_k(retrieved, expected, 5) == 1.0
    assert hit_rate_at_k(retrieved, expected, 1) == 1.0


def test_miss() -> None:
    retrieved = ["Art. 1", "Art. 2"]
    expected = {"Art. 54"}
    assert recall_at_k(retrieved, expected, 5) == 0.0
    assert mrr(retrieved, expected) == 0.0
    assert hit_rate_at_k(retrieved, expected, 5) == 0.0


def test_partial_and_rank_sensitivity() -> None:
    expected = {"Art. 54"}
    first = ["Art. 54", "Art. 1"]
    third = ["Art. 1", "Art. 2", "Art. 54"]
    assert mrr(first, expected) == 1.0
    assert mrr(third, expected) == 1.0 / 3
    assert recall_at_k(third, expected, 2) == 0.0
    assert recall_at_k(third, expected, 3) == 1.0
    assert precision_at_k(first, expected, 2) == 0.5
