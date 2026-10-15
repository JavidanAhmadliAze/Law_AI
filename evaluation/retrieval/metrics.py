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


def ndcg_at_k(
    retrieved: list[str],
    core: set[str],
    k: int,
    supporting: set[str] | None = None,
) -> float:
    """Graded nDCG@k with the standard exponential gain, 2**rel - 1.

    Three grades, matching the labels the golden datasets carry:

        core        rel 2  ->  gain 3   the provision the answer must quote
        supporting  rel 1  ->  gain 1   needed to apply it, not the rule itself
        anything else      ->  gain 0

    The exponential gain is the point of grading it this way: a core provision is
    worth three supporting ones, so a run that leads with the rule beats one that
    leads with three provisions about the rule. Under the binary version this
    metric replaces, every match scored 1 and `supporting` was passed as nothing
    at all — so delivering the surrounding chapter instead of the provision cost
    nothing, which is exactly the failure the one-article rule was built to stop.

    `supporting` is optional so the binary behaviour (core only) is still
    reachable, and callers that never had a supporting set keep their numbers
    comparable.
    """
    supporting = supporting or set()

    def gain(article: str) -> float:
        if article in core:
            return 3.0  # 2**2 - 1
        return 1.0 if article in supporting else 0.0  # 2**1 - 1

    dcg = sum(
        gain(article) / math.log2(rank + 1) for rank, article in enumerate(retrieved[:k], start=1)
    )
    # The ideal ranking puts every core provision first, then the supporting
    # ones, and is cut at k like the actual ranking.
    ideal_gains = sorted([3.0] * len(core) + [1.0] * len(supporting - core), reverse=True)[:k]
    ideal = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_gains, start=1))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision(retrieved: list[str], expected: set[str]) -> float:
    """Mean of precision@i taken at each rank where a relevant item appears.

    Rewards putting relevant items early in a way recall@k cannot: two runs with
    identical recall@10 score differently if one front-loads the hits. Averaged
    over queries this is MAP.
    """
    if not expected:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in expected:
            hits += 1
            total += hits / rank
    return total / min(len(expected), len(retrieved)) if retrieved else 0.0


def f1_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    p = precision_at_k(retrieved, expected, k)
    r = recall_at_k(retrieved, expected, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def recall(retrieved: list[str], expected: set[str]) -> float:
    """Recall over the WHOLE returned list — the stage-2 question.

    Stage 2 returns a variable number of chunks, so recall@k with a fixed k
    would penalise a run for being selective rather than for missing anything.
    """
    if not expected:
        return 0.0
    return len(set(retrieved) & expected) / len(expected)


def precision(retrieved: list[str], relevant: set[str]) -> float:
    """Precision over the whole returned list. `relevant` should be core +
    supporting: a supporting chunk is not noise, so counting it against the
    retriever would understate precision."""
    if not retrieved:
        return 0.0
    return len(set(retrieved) & relevant) / len(retrieved)
