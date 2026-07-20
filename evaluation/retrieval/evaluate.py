"""Retrieval evaluation — runs golden questions through the search service.

Mirrors the online pipeline: English questions are translated with the same
translator production uses (LLM when configured, glossary fallback otherwise)
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
from law_ai.services.translation.base import BaseTranslator, Direction
from law_ai.services.translation.client import GlossaryOnlyTranslator


def _make_translator(settings: Any) -> tuple[BaseTranslator, str]:
    """Production translator when the LLM is configured, glossary fallback."""
    try:
        from law_ai.services.llm.factory import create_llm
        from law_ai.services.translation.factory import create_translator

        llm = create_llm(settings)
        return create_translator(settings, llm=llm), "llm"
    except Exception:
        return GlossaryOnlyTranslator(), "glossary-only"


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
    translator, translator_kind = _make_translator(settings)
    await search.startup()

    rows = [
        json.loads(line)
        for line in Path(config.dataset_path).read_text().splitlines()
        if line.strip()
    ]

    per_query: list[dict[str, Any]] = []
    try:
        for row in rows:
            query = row["question"]
            if row.get("language") == "en":
                query = await translator.translate(query, Direction.EN_TO_PL)
            results = await search.retrieve(query, top_k=max(config.k_values))
            retrieved = _dedup_articles(results)
            expected = set(row["expected_articles"])
            per_query.append(
                {
                    "question": row["question"],
                    "translated_query": query,
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
        "translator": translator_kind,
        "mrr": round(mean(q["mrr"] for q in per_query), 4),
        **{f"hit@{k}": round(mean(q[f"hit@{k}"] for q in per_query), 4) for k in config.k_values},
        **{
            f"recall@{k}": round(mean(q[f"recall@{k}"] for q in per_query), 4)
            for k in config.k_values
        },
        **{f"ndcg@{k}": round(mean(q[f"ndcg@{k}"] for q in per_query), 4) for k in config.k_values},
    }
    return {"summary": summary, "per_query": per_query}
