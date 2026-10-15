"""Retrieval evaluation — runs golden questions through the search service.

Mirrors the online pipeline: English questions are translated with the same
the query as written — there is no translation step any more
before retrieval, so the cross-lingual path is what actually gets measured.

Articles are matched act-qualified ("kodeks-cywilny:Art. 691") — bare article
numbers repeat across acts. Retrieved chunks are deduplicated to article
level (first occurrence keeps its rank) since several chunks of one article
count as one relevant hit.
"""

import json
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.config import EvalConfig
from evaluation.retrieval import metrics
from law_ai.dependencies import get_settings
from law_ai.services.embedding.factory import create_embedder
from law_ai.services.opensearch.factory import create_search_service


def _dedup_articles(chunks: list[Any]) -> list[str]:
    seen: list[str] = []
    for chunk in chunks:
        key = f"{chunk.chunk.metadata.act}:{chunk.chunk.metadata.article}"
        if key not in seen:
            seen.append(key)
    return seen


async def evaluate_retrieval(config: EvalConfig) -> dict[str, Any]:
    settings = get_settings()
    embedder = create_embedder(settings)
    search = create_search_service(settings, embedder)
    await search.startup()

    rows = [
        json.loads(line)
        for line in Path(config.dataset_path).read_text().splitlines()
        if line.strip()
    ]

    per_query: list[dict[str, Any]] = []
    try:
        for row in rows:
            # As written, in whatever language the dataset holds. Nothing
            # translates it: the pipeline's rewriter emits Polish sub-queries
            # directly, so an EN->PL hop no longer exists to mirror here.
            query = row["question"]
            results = await search.retrieve(query, top_k=max(config.k_values))
            retrieved = _dedup_articles(results)
            expected = set(row["expected_articles"])
            per_query.append(
                {
                    "question": row["question"],
                    "query": query,
                    "retrieved": retrieved,
                    "expected": sorted(expected),
                    "mrr": metrics.mrr(retrieved, expected),
                    **{
                        f"recall@{k}": metrics.recall_at_k(retrieved, expected, k)
                        for k in config.k_values
                    },
                    **{
                        f"hit@{k}": metrics.hit_rate_at_k(retrieved, expected, k)
                        for k in config.k_values
                    },
                    **{
                        f"precision@{k}": metrics.precision_at_k(retrieved, expected, k)
                        for k in config.k_values
                    },
                    **{
                        f"ndcg@{k}": metrics.ndcg_at_k(retrieved, expected, k)
                        for k in config.k_values
                    },
                }
            )
    finally:
        await search.teardown()

    summary = {
        "queries": len(per_query),
        "mrr": round(mean(q["mrr"] for q in per_query), 4),
        **{f"hit@{k}": round(mean(q[f"hit@{k}"] for q in per_query), 4) for k in config.k_values},
        **{
            f"recall@{k}": round(mean(q[f"recall@{k}"] for q in per_query), 4)
            for k in config.k_values
        },
        **{f"ndcg@{k}": round(mean(q[f"ndcg@{k}"] for q in per_query), 4) for k in config.k_values},
    }
    return {"summary": summary, "per_query": per_query}
