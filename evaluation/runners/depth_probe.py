"""How deep would retrieval have to go to hold every core provision?

The two-stage report says a provision was "never retrieved", but not why. Three
very different things produce that line, and they need opposite fixes:

  rank <= per-leg depth   it WAS reachable; fusion or truncation lost it
  rank just beyond it     the per-leg candidate window is the binding limit
  rank in the hundreds    the query wording never points at it; depth is no cure

So this runs each leg ALONE, to a depth far past what retrieval uses, and reports
the best rank each labelled core provision achieves on any leg of any sub-query.
No RRF and no reranker: both operate on what the legs already returned, so
neither can explain a provision that no leg ranked.

    uv run python -m evaluation.runners.depth_probe [dataset.jsonl]
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from opensearchpy import AsyncOpenSearch

from law_ai.dependencies import get_settings
from law_ai.services.agents.context import AgentServices
from law_ai.services.agents.nodes.query_rewriter import query_rewriter
from law_ai.services.embedding.factory import create_embedder
from law_ai.services.llm.factory import create_fast_llm, create_llm, create_reasoning_llm
from law_ai.services.opensearch.factory import create_search_service
from law_ai.services.reranker.factory import create_reranker

_PROBE_DEPTH = 300
_LIVE_DEPTH = 25  # _CANDIDATES_PER_LEG — what each leg actually fetches today


async def _leg_ranks(
    client: AsyncOpenSearch, index: str, body: dict[str, Any], wanted: set[str]
) -> dict[str, int]:
    res = await client.search(index=index, body=body)
    return {
        h["_source"]["chunk_id"]: i
        for i, h in enumerate(res["hits"]["hits"], 1)
        if h["_source"]["chunk_id"] in wanted
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="evaluation/datasets/golden_tax.jsonl")
    args = parser.parse_args()

    settings = get_settings()
    fast = create_fast_llm(settings)
    search = create_search_service(settings, create_embedder(settings))
    await search.startup()
    embedder = create_embedder(settings)
    services = AgentServices(
        llm=create_llm(settings),
        fast_llm=fast,
        reasoning_llm=create_reasoning_llm(settings),
        search=search,
        reranker=create_reranker(settings),
        decompose_queries=settings.agent.decompose_queries,
        score_floor=settings.agent.score_floor,
        max_chunks=settings.agent.max_chunks,
        rerank_candidates_size=settings.agent.rerank_candidates,
        max_searches=settings.agent.max_searches,
    )
    config = {"configurable": {"services": services}}
    client = AsyncOpenSearch(
        hosts=[{"host": settings.opensearch.host, "port": settings.opensearch.port}]
    )
    index = settings.opensearch.index
    rows = [json.loads(x) for x in Path(args.dataset).read_text().splitlines() if x.strip()]

    best: dict[str, tuple[int, str, int, str]] = {}  # chunk_id -> (rank, leg, qid, article)
    try:
        for row in rows:
            core = {c["chunk_id"]: c["article"] for c in row["core"]}
            rewritten = await query_rewriter(
                {"messages": [HumanMessage(content=row.get("query_en") or row["query"])]},
                config,  # type: ignore[arg-type]
            )
            for query in rewritten["search_queries_pl"]:
                vector = await embedder.embed_query(query)
                knn: dict[str, Any] = {"vector": vector, "k": _PROBE_DEPTH}
                match: dict[str, Any] = {"match": {"text": query}}
                bodies = {
                    "dense": {
                        "size": _PROBE_DEPTH,
                        "_source": ["chunk_id"],
                        "query": {"knn": {"vector": knn}},
                    },
                    "bm25": {
                        "size": _PROBE_DEPTH,
                        "_source": ["chunk_id"],
                        "query": (match),
                    },
                }
                for leg, body in bodies.items():
                    for chunk_id, rank in (
                        await _leg_ranks(client, index, body, set(core))
                    ).items():
                        prior = best.get(chunk_id)
                        if prior is None or rank < prior[0]:
                            best[chunk_id] = (rank, leg, row["id"], core[chunk_id])
            print(
                f"  probed q{row['id']} ({len(rewritten['search_queries_pl'])} searches)",
                flush=True,
            )

        total = sum(len(r["core"]) for r in rows)
        buckets: Counter[str] = Counter()
        detail: list[tuple[int, int, str, str]] = []
        for row in rows:
            for c in row["core"]:
                hit = best.get(c["chunk_id"])
                if hit is None:
                    buckets[f"unreachable (no leg ranks it in {_PROBE_DEPTH})"] += 1
                    detail.append((10**6, row["id"], c["article"], "-"))
                    continue
                rank, leg, _, _ = hit
                if rank <= _LIVE_DEPTH:
                    buckets[f"already in the window (rank <= {_LIVE_DEPTH})"] += 1
                elif rank <= 50:
                    buckets["would need depth 50"] += 1
                elif rank <= 100:
                    buckets["would need depth 100"] += 1
                else:
                    buckets[f"would need depth > 100 (<= {_PROBE_DEPTH})"] += 1
                detail.append((rank, row["id"], c["article"], leg))

        print(f"\n===== best rank achieved by ANY leg, {total} core provisions =====")
        order = [
            f"already in the window (rank <= {_LIVE_DEPTH})",
            "would need depth 50",
            "would need depth 100",
            f"would need depth > 100 (<= {_PROBE_DEPTH})",
            f"unreachable (no leg ranks it in {_PROBE_DEPTH})",
        ]
        for name in order:
            n = buckets.get(name, 0)
            print(f"  {name:<46}{n:>4}   {n / total:6.1%}")

        print(f"\n----- every core provision ranked past {_LIVE_DEPTH} -----")
        for rank, qid, article, leg in sorted(detail, reverse=True):
            if rank > _LIVE_DEPTH:
                shown = f">{_PROBE_DEPTH}" if rank == 10**6 else str(rank)
                print(f"  q{qid:<4} {article:<18} best rank {shown:<6} on {leg}")
    finally:
        await client.close()
        await search.teardown()


if __name__ == "__main__":
    asyncio.run(main())
