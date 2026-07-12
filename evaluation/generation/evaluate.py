"""Generation evaluation — runs golden questions through the FULL agent graph
and judges the answers (LLM-as-judge + deterministic citation accuracy)."""

import json
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.config import EvalConfig
from evaluation.generation.metrics import citation_accuracy, judge_answer
from law_ai.dependencies import get_settings
from law_ai.services.agents.factory import AgentGraphFactory
from law_ai.services.embedding.factory import EmbedderFactory
from law_ai.services.llm.factory import LLMFactory
from law_ai.services.opensearch.factory import SearchServiceFactory
from law_ai.services.translation.factory import TranslatorFactory


async def evaluate_generation(config: EvalConfig) -> dict[str, Any]:
    settings = get_settings()
    llm = LLMFactory.create(settings)
    embedder = EmbedderFactory.create(settings)
    search = SearchServiceFactory.create(settings, embedder)
    translator = TranslatorFactory.create(settings, llm=llm)
    await search.startup()

    graph = AgentGraphFactory.create(llm=llm, search=search, translator=translator)

    rows = [
        json.loads(line)
        for line in Path(config.dataset_path).read_text().splitlines()
        if line.strip()
    ]

    per_query: list[dict[str, Any]] = []
    try:
        for row in rows:
            result = await graph.ainvoke({"question": row["question"], "history": []})
            final = result.get("final_answer")
            answer_text = final.answer if final else ""
            cited = [c.article for c in final.citations] if final else []

            scores = await judge_answer(
                llm,
                question=row["question"],
                reference=row["reference_answer"],
                answer=answer_text,
            )
            per_query.append(
                {
                    "question": row["question"],
                    "answer": answer_text,
                    "faithfulness": scores.faithfulness,
                    "relevance": scores.relevance,
                    "correctness": scores.correctness,
                    "citation_accuracy": citation_accuracy(cited, row["expected_articles"]),
                    "comment": scores.comment,
                }
            )
    finally:
        await search.teardown()

    summary = {
        "queries": len(per_query),
        **{
            metric: round(mean(q[metric] for q in per_query), 4)
            for metric in ("faithfulness", "relevance", "correctness", "citation_accuracy")
        },
    }
    return {"summary": summary, "per_query": per_query}
