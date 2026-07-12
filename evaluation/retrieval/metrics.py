"""Deterministic retrieval metrics — no LLM involved.

All metrics take the ranked list of retrieved article ids and the set of
expected (relevant) article ids for one query.
"""

import math


def hit_rate_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """1.0 if any relevant article appears in the top-k."""
    return 1.0 if set(retrieved[:k]) & expected else 0.0


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(retrieved[:k]) & expected) / len(expected)


def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & expected) / k


def mrr(retrieved: list[str], expected: set[str]) -> float:
    for rank, article in enumerate(retrieved, start=1):
        if article in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, article in enumerate(retrieved[:k], start=1)
        if article in expected
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0
