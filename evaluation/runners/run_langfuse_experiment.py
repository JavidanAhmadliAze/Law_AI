"""Retrieval evaluation as a Langfuse experiment.

    uv run python -m evaluation.runners.run_langfuse_experiment

Unlike run_eval (local JSON report), this pushes everything to Langfuse:
- golden questions become items of the "civil-retrieval" dataset,
- each run is a named experiment run — one trace per question, with the
  service's "retrieval" span (chunk-level RRF/rerank scores) nested inside,
- hit@5 / recall@5 / mrr / ndcg@5 attach to each trace as scores.

Compare runs in the UI under Datasets → civil-retrieval → Runs (e.g. when
swapping embedding/reranker models).
"""

import asyncio
import contextlib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from langfuse.experiment import Evaluation

from evaluation.config import EvalConfig
from evaluation.retrieval import metrics
from law_ai.dependencies import get_settings
from law_ai.services.embedding.factory import create_embedder
from law_ai.services.langfuse.factory import create_langfuse
from law_ai.services.llm.factory import create_llm
from law_ai.services.opensearch.factory import create_search_service
from law_ai.services.translation.base import Direction
from law_ai.services.translation.factory import create_translator

DATASET_NAME = "civil-retrieval"
_K = 5


def _sync_dataset(client: Any, rows: list[dict]) -> None:
    """Upsert golden rows as dataset items (stable ids → idempotent)."""
    client.create_dataset(
        name=DATASET_NAME,
        description="Civil-law retrieval golden set (EN questions, act-qualified articles)",
    )
    for row in rows:
        item_id = hashlib.sha256(row["question"].encode()).hexdigest()[:16]
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=item_id,
            input={"question": row["question"], "language": row["language"]},
            expected_output={"articles": row["expected_articles"]},
            metadata={"acts": sorted({a.split(":")[0] for a in row["expected_articles"]})},
        )


def _evaluate(*, input: Any, output: Any, expected_output: Any, **kwargs: Any) -> list[Evaluation]:
    retrieved = output["retrieved"]
    expected = set(expected_output["articles"])
    return [
        Evaluation(name="hit@5", value=metrics.hit_rate_at_k(retrieved, expected, _K)),
        Evaluation(name="recall@5", value=metrics.recall_at_k(retrieved, expected, _K)),
        Evaluation(name="precision@5", value=metrics.precision_at_k(retrieved, expected, _K)),
        Evaluation(name="mrr", value=metrics.mrr(retrieved, expected)),
        Evaluation(name="ndcg@5", value=metrics.ndcg_at_k(retrieved, expected, _K)),
    ]


def main() -> None:
    config = EvalConfig()
    settings = get_settings()
    if not settings.langfuse.enabled:
        raise SystemExit("LANGFUSE__ENABLED=false — this runner needs Langfuse")

    tracer = create_langfuse(settings)
    tracer.callback_handler()  # initializes the global Langfuse client
    from langfuse import get_client

    client = get_client()

    rows = [
        json.loads(line)
        for line in Path(config.dataset_path).read_text().splitlines()
        if line.strip()
    ]
    _sync_dataset(client, rows)

    # run_experiment drives its own event loop (run_async_safely), so all
    # async clients must be created INSIDE that loop — lazy init on first task,
    # locked so concurrent first tasks don't double-load the reranker
    services: dict[str, Any] = {}
    init_lock = asyncio.Lock()

    async def _ensure_services() -> dict[str, Any]:
        async with init_lock:
            if not services:
                embedder = create_embedder(settings)
                search = create_search_service(settings, embedder, tracer=tracer)
                await search.startup()
                services["search"] = search
                services["translator"] = create_translator(settings, llm=create_llm(settings))
        return services

    async def task(*, item: Any, **kwargs: Any) -> dict:
        svc = await _ensure_services()
        question = item.input["question"]
        query = question
        if item.input.get("language") == "en":
            query = await svc["translator"].translate(query, Direction.EN_TO_PL)
        results = await svc["search"].retrieve(query, top_k=_K)
        retrieved: list[str] = []
        for r in results:
            key = f"{r.chunk.metadata.act}:{r.chunk.metadata.article}"
            if key not in retrieved:
                retrieved.append(key)
        return {"translated_query": query, "retrieved": retrieved}

    dataset = client.get_dataset(DATASET_NAME)
    run_name = (
        f"bge-m3+{settings.reranker.model.split('/')[-1] or 'no-rerank'}"
        f"-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    try:
        result = dataset.run_experiment(
            name="civil-retrieval",
            run_name=run_name,
            description="EN→PL LLM translation, hybrid RRF, cross-encoder rerank",
            task=task,
            evaluators=[_evaluate],
            max_concurrency=2,  # CPU reranker — don't thrash it
        )
    finally:
        if "search" in services:  # best-effort close from a fresh loop
            with contextlib.suppress(Exception):
                asyncio.run(services["search"].teardown())
        tracer.shutdown()

    report = _write_report(result, run_name, config)
    print(f"run: {run_name} — {len(result.item_results)} items")
    print(f"summary: {json.dumps(report['summary'], ensure_ascii=False)}")
    print(f"report: {report['path']}")
    print("View: Langfuse UI → Datasets → civil-retrieval → Runs")


def _write_report(result: Any, run_name: str, config: EvalConfig) -> dict[str, Any]:
    """Aggregate per-item scores into a JSON report file next to run_eval's."""
    per_item: list[dict[str, Any]] = []
    for item_result in result.item_results:
        scores = {e.name: e.value for e in (item_result.evaluations or [])}
        item = item_result.item
        question = item.input.get("question") if isinstance(item.input, dict) else str(item.input)
        output = item_result.output or {}
        per_item.append(
            {
                "question": question,
                "translated_query": output.get("translated_query", ""),
                "retrieved": output.get("retrieved", []),
                "expected": (item.expected_output or {}).get("articles", []),
                **scores,
            }
        )
    score_names = sorted(
        {k for row in per_item for k in row if k.startswith(("hit", "recall", "mrr", "ndcg"))}
    )
    summary = {
        "run_name": run_name,
        "items": len(per_item),
        **{name: round(mean(row.get(name, 0.0) for row in per_item), 4) for name in score_names},
    }
    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"langfuse-{run_name}.json"
    path.write_text(
        json.dumps({"summary": summary, "per_item": per_item}, indent=2, ensure_ascii=False)
    )
    return {"summary": summary, "path": str(path)}


if __name__ == "__main__":
    main()
