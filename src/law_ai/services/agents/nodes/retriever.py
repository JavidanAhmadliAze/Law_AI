"""Retriever — deterministic, and the only place the corpus is read.

One forward pass, no loop: the searches were decided by the planning
subgraph, so they are all known here and go out concurrently.

    retrieve (act-scoped, hybrid)  ->  rerank per sub-query  ->  merge  ->  writer

Each pool is reranked against the query that produced it, not against a merged
brief: cross-encoder scores are query-conditional, so judging a bankruptcy
passage against a mostly-employment brief scores it low and drops it.

Nothing filters between the reranker and the writer. Every gate that has sat
here — corroboration by article family, the per-article score ratio, the
score-gap cut and its per-search floor, a per-search window, and the rule
confining a sub-query to the article of its rank-1 chunk — was tried and removed,
each because it discarded provisions the cross-encoder had already found. What
reaches the writer is the top `max_chunks` by rerank score.
"""

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig

from law_ai.schemas.chunk import RetrievedChunk
from law_ai.services.agents.context import services_from_config
from law_ai.services.agents.state import AgentState


def _format(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{c.chunk.metadata.article} — {c.chunk.metadata.act}]\n{c.chunk.text}" for c in chunks
    )


async def retriever(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    # sub-query -> the ONE act it is confined to. "" is the whole corpus: the
    # planner leaves the act out when it cannot place the search, and a search
    # against everything costs precision where an empty filter would return
    # nothing at all.
    searches: dict[str, str] = state["search_queries_pl"]
    query_act_pairs = list(searches.items())

    pools = await asyncio.gather(
        *(
            services.search.retrieve(
                q,
                top_k=services.rerank_candidates_size,
                act=a,
            )
            for q, a in query_act_pairs
        )
    )

    reranker = services.reranker
    if reranker is None:
        raise RuntimeError("retriever needs a reranker in the agent services")
    ranked_per_query = await asyncio.gather(
        *(reranker.rerank(q, pool) for (q, _), pool in zip(query_act_pairs, pools, strict=True))
    )
    # Normalise each sub-query's scores by its own best, so the merge compares
    # like with like. The cross-encoder returns unbounded logits and they are
    # query-conditional: one sub-query's best hit scores 11 while another's
    # scores 3, and sorting the raw numbers lets the first sub-query take every
    # slot regardless of what the second found.
    #
    # The floor is what stops that becoming the opposite error. Dividing by the
    # sub-query's own max alone would rescale a search that found nothing —
    # top score 1.2 — so that its best chunk also reaches 1.0 and competes with
    # a genuine hit. Below the floor a sub-query is judged against the floor
    # instead, and stays as weak as it deserves.
    floor = services.score_floor
    normalised: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for ranked in ranked_per_query:
        if not ranked:
            continue
        denom = max(max(c.score for c in ranked), floor)
        for chunk in ranked:
            value = chunk.score / denom
            cid = chunk.chunk.chunk_id
            if cid not in normalised or value > normalised[cid]:
                normalised[cid] = value
                best[cid] = chunk

    order = sorted(best.values(), key=lambda c: -normalised[c.chunk.chunk_id])
    chunks = order[: services.max_chunks]
    # Optional delivery cut (AGENT__DELIVERY_SCORE_CUT): keep what scores at
    # least the cut, but never fewer than one chunk per sub-query (floored at
    # delivery_min_chunks) — a provision whose score is merely lower than its
    # siblings' is still the one its sub-query was planned to find.
    if services.delivery_score_cut > 0:
        floor = max(services.delivery_min_chunks, len(query_act_pairs))
        chunks = [
            c
            for i, c in enumerate(chunks)
            if normalised[c.chunk.chunk_id] >= services.delivery_score_cut or i < floor
        ]

    chunks.sort(key=lambda c: (c.chunk.metadata.act, c.chunk.metadata.article))

    return {
        "notes": [_format(chunks)] if chunks else [],
        "retrieved": [
            {
                "article": c.chunk.metadata.article,
                "act": c.chunk.metadata.act,
                "chunk_id": c.chunk.chunk_id,
                "score": round(c.score, 5),
                "normalised": round(normalised[c.chunk.chunk_id], 5),
            }
            for c in chunks
        ],
    }
