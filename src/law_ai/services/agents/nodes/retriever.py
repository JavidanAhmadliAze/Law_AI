"""Retriever — deterministic, and the only place the corpus is read.

One forward pass, no loop: the searches were decided by the query rewriter, so
they are all known here and go out concurrently.

    retrieve (+ metadata filter)  ->  rerank  ->  top-k chunks  ->  writer

Each pool is reranked against the query that produced it, not against a merged
brief: cross-encoder scores are query-conditional, so judging a bankruptcy
passage against a mostly-employment brief scores it low and drops it.
"""

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig

from law_ai.schemas.chunk import RetrievedChunk
from law_ai.services.agents.context import services_from_config
from law_ai.services.agents.state import AgentOutputState


def _format(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{c.chunk.metadata.article} — {c.chunk.metadata.act}]\n{c.chunk.text}" for c in chunks
    )


async def retriever(state: AgentOutputState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    # already Polish and glossary-normalised, and guaranteed non-empty — how many
    # searches a question needs was settled by the rewriter
    queries = state["search_queries_pl"]
    # narrows the candidate space before scoring: `act`, `domain`, `article` and
    # `effective_date` are indexed keyword fields, so a filter is free at search
    # time and removes whole acts the question cannot be about
    filters = state.get("metadata_filters") or None

    # the budget is what costs (~0.9s per reranked pair), so it is split across
    # the searches rather than capping how many issues a question may have
    per_query = max(services.min_candidates_per_query, services.rerank_budget // len(queries))
    pools = await asyncio.gather(
        *(services.search.retrieve(q, top_k=per_query, filters=filters) for q in queries)
    )
    ranked_per_query = await asyncio.gather(
        *(services.search.rerank(q, pool) for q, pool in zip(queries, pools, strict=True))
    )

    # Each sub-query already ranked its own pool, so there is no final global
    # sort: cross-encoder scores are query-conditional and simply not comparable
    # between queries — measured, one sub-query's best scored 0.95 while
    # another's best scored 0.49, and merging them by raw value let the first
    # take nearly every slot. Each sub-query gets its own share instead, so an
    # issue that happens to score low still reaches the writer.
    share = max(1, services.retriever_top_k // len(queries))
    seen: set[str] = set()
    top: list[RetrievedChunk] = []
    for ranked in ranked_per_query:
        taken = 0
        for c in ranked:
            if c.chunk.chunk_id in seen:
                continue
            seen.add(c.chunk.chunk_id)
            top.append(c)
            taken += 1
            if taken >= share:
                break

    # Grouped by article rather than ordered by score. The chunks carry their own
    # paragraph markers ("Art. 484. § 1.", "§ 2.") so the writer can order within
    # an article itself — what it cannot do is recognise two blocks under
    # identical [Art. 484] headers as one provision when rank scatters them apart.
    # That produced an answer reporting § 1 as missing while it sat seven blocks
    # further down.
    top.sort(key=lambda c: (c.chunk.metadata.act, c.chunk.metadata.article))

    return {
        "notes": [_format(top)] if top else [],
        "retrieved": [
            {
                "article": c.chunk.metadata.article,
                "act": c.chunk.metadata.act,
                "chunk_id": c.chunk.chunk_id,
                "score": round(c.score, 5),
            }
            for c in top
        ],
    }
