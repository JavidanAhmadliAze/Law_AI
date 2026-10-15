"""What reaches the writer: one rerank pass per sub-query, merged, capped.

Every selection gate between the reranker and the writer has been removed, so
the merge IS the selection. These pin the two things it still has to get right:
how a chunk several searches found is credited, and what the ordering means.
"""

from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig

from law_ai.schemas.chunk import ChunkMetadata, LawChunk, RetrievedChunk
from law_ai.services.agents.context import AgentServices
from law_ai.services.agents.nodes.retriever import retriever


def _c(name: str, score: float, act: str = "a", article: str | None = None) -> RetrievedChunk:
    """`article` defaults to the chunk id, so a pool of distinct ids is a pool
    of distinct articles unless a test groups them on purpose."""
    return RetrievedChunk(
        chunk=LawChunk(
            chunk_id=name,
            text=name,
            metadata=ChunkMetadata(article=article or name, act=act, domain="tax"),
        ),
        score=score,
    )


class _Search:
    """Returns a canned pool per query, and records the act scope it was given."""

    def __init__(self, pools: dict[str, list[RetrievedChunk]]) -> None:
        self.pools = pools
        self.acts: list[str | None] = []

    async def retrieve(
        self, query: str, *, top_k: int = 5, act: str | None = None
    ) -> list[RetrievedChunk]:
        self.acts.append(act)
        return self.pools[query][:top_k]


class _Reranker:
    """Scores by a per-query table, so each pool is judged against its own query."""

    def __init__(self, table: dict[str, dict[str, float]]) -> None:
        self.table = table
        self.queries: list[str] = []

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        self.queries.append(query)
        scored = [
            _c(
                c.chunk.chunk_id,
                self.table[query][c.chunk.chunk_id],
                article=c.chunk.metadata.article,
            )
            for c in chunks
        ]
        return sorted(scored, key=lambda c: -c.score)


def _services(search: Any, reranker: Any, **kw: Any) -> AgentServices:
    """AgentServices carries no tunable defaults — a default there would be a
    second copy of the setting — so a test states the ones it depends on and
    overrides the rest through `kw`."""
    stub = cast(Any, object())
    tunables: dict[str, Any] = {
        "decompose_queries": True,
        "score_floor": 4.0,
        "max_chunks": 10,
        "rerank_candidates_size": 50,
        "max_searches": 6,
    }
    return AgentServices(
        llm=stub,
        fast_llm=stub,
        reasoning_llm=stub,
        search=search,
        reranker=reranker,
        **(tunables | kw),
    )


def _state(queries: list[str], acts: dict[str, str] | None = None) -> Any:
    """search_queries_pl maps each sub-query to the ONE act it is confined to;
    "" is the whole corpus, which is what the planner leaves behind when it
    cannot place a search."""
    acts = acts or {}
    return {"search_queries_pl": {q: acts.get(q, "") for q in queries}, "messages": []}


async def _run(state: Any, services: AgentServices) -> list[str]:
    """What reaches the writer."""
    config = cast(RunnableConfig, {"configurable": {"services": services}})
    out = await retriever(state, config)
    return [r["chunk_id"] for r in out["retrieved"]]


@pytest.mark.asyncio
async def test_each_sub_query_is_reranked_against_itself() -> None:
    """Not against a merged brief: a chunk answering search 2 must not be scored
    against search 1's wording, which is what collapsed cross-encoder scores
    when a bankruptcy passage met a mostly-employment brief."""
    search = _Search({"q1": [_c("x", 0.1)], "q2": [_c("y", 0.1)]})
    reranker = _Reranker({"q1": {"x": 0.9}, "q2": {"y": 0.8}})
    services = _services(search, reranker)
    await _run(_state(["q1", "q2"], {"q1": "ord", "q2": "ord"}), services)
    assert reranker.queries == ["q1", "q2"]


@pytest.mark.asyncio
async def test_search_is_confined_to_the_acts_the_planner_named() -> None:
    search = _Search({"q1": [_c("x", 0.1)], "q2": [_c("y", 0.1)]})
    reranker = _Reranker({"q1": {"x": 0.9}, "q2": {"y": 0.8}})
    services = _services(search, reranker)
    await _run(_state(["q1", "q2"], {"q1": "ord"}), services)
    # A search with no act searches the whole corpus, not an empty corpus.
    assert search.acts == ["ord", ""]  # "" = the whole corpus


@pytest.mark.asyncio
async def test_a_chunk_two_searches_found_carries_its_highest_score() -> None:
    """Art. 70 § 7 was rank 3 / 0.952 in one search and rank 5 / 0.974 in
    another. Keeping the first occurrence understates what the cross-encoder
    made of it — and with one slot left, that decides whether it ships.

    The returned list is sorted by act and article for the writer, so the merge
    is observable only through WHICH chunks clear the cap, never their order.
    """
    search = _Search({"q1": [_c("shared", 0.1), _c("rival", 0.1)], "q2": [_c("shared", 0.1)]})
    reranker = _Reranker({"q1": {"shared": 0.952, "rival": 0.960}, "q2": {"shared": 0.974}})
    services = _services(search, reranker, max_chunks=1)
    # Judged on its 0.952 showing it loses to rival's 0.960; on its best, it wins.
    assert await _run(_state(["q1", "q2"]), services) == ["shared"]


@pytest.mark.asyncio
async def test_scores_are_normalised_per_sub_query() -> None:
    """Cross-encoder scores are query-conditional and unbounded, so raw numbers
    are not comparable between sub-queries. q2's best hit scores 12 and q1's
    scores 5; without normalisation q2 takes both slots even though q1's chunk
    is its own search's clear winner. Divided by each sub-query's own best, q1's
    5/5 = 1.0 beats q2's second at 6/12 = 0.5."""
    pools = {"q1": [_c("q1_best", 0.1)], "q2": [_c("q2_best", 0.1), _c("q2_second", 0.1)]}
    table = {"q1": {"q1_best": 5.0}, "q2": {"q2_best": 12.0, "q2_second": 6.0}}
    services = _services(_Search(pools), _Reranker(table), max_chunks=2)
    delivered = await _run(_state(["q1", "q2"]), services)
    assert set(delivered) == {"q1_best", "q2_best"}  # q2_second loses to q1_best


@pytest.mark.asyncio
async def test_the_floor_stops_a_weak_sub_query_normalising_itself_up() -> None:
    """A sub-query whose best hit scores 1.2 must not have that chunk rescaled
    to 1.0 and placed level with a genuine hit. Below the floor it is divided by
    the floor instead: 1.2/4.0 = 0.3, well behind the strong search's 1.0."""
    pools = {"q1": [_c("strong", 0.1)], "q2": [_c("weak", 0.1)]}
    table = {"q1": {"strong": 8.0}, "q2": {"weak": 1.2}}
    services = _services(_Search(pools), _Reranker(table), max_chunks=1)
    assert await _run(_state(["q1", "q2"]), services) == ["strong"]


@pytest.mark.asyncio
async def test_a_search_delivers_its_window_whatever_the_articles() -> None:
    """No article test any more. Confining a sub-query to the article of its
    rank-1 chunk discarded provisions the cross-encoder had already found: on the
    board-member question eight sub-queries produced a complete pool and the rule
    delivered half of it, art. 108 § 4 going out from rank 2 at 4.015 because
    rank 1 belonged to art. 116."""
    pool = [_c(n, 0.1) for n in ("a", "b", "c")]  # three distinct articles
    reranker = _Reranker({"q1": {"a": 0.99, "b": 0.98, "c": 0.97}})
    services = _services(_Search({"q1": pool}), reranker, max_chunks=10)
    assert await _run(_state(["q1"]), services) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_every_reranked_chunk_is_eligible() -> None:
    """No per-search window: the cap is the only cut. A sub-query contributes as
    much of its reranked list as the merged cap has room for."""
    pool = [_c(f"p{i}", 0.1, article="Art. 70") for i in range(6)]
    reranker = _Reranker({"q1": {f"p{i}": 0.99 - i / 100 for i in range(6)}})
    services = _services(_Search({"q1": pool}), reranker, max_chunks=10)
    assert await _run(_state(["q1"]), services) == [f"p{i}" for i in range(6)]


@pytest.mark.asyncio
async def test_the_cap_still_bounds_the_merged_result() -> None:
    """Two searches offering two articles' worth of chunks, capped at two."""
    pools = {
        "q1": [_c("a1", 0.1, article="A"), _c("a2", 0.1, article="A")],
        "q2": [_c("b1", 0.1, article="B"), _c("b2", 0.1, article="B")],
    }
    table = {
        "q1": {"a1": 0.99, "a2": 0.98},
        "q2": {"b1": 0.97, "b2": 0.96},
    }
    services = _services(_Search(pools), _Reranker(table), max_chunks=2)
    assert await _run(_state(["q1", "q2"]), services) == ["a1", "a2"]


@pytest.mark.asyncio
async def test_delivery_cut_keeps_the_scored_and_one_per_sub_query() -> None:
    """With the cut on, chunks below it go unless they fill the floor of one per
    sub-query; with it off (default) the top max_chunks are delivered uncut."""
    pool = [_c("a", 1.0), _c("b", 1.0), _c("c", 1.0), _c("d", 1.0)]
    search = _Search({"q1": pool, "q2": pool})
    reranker = _Reranker(
        {
            "q1": {"a": 10.0, "b": 9.5, "c": 4.0, "d": 3.0},
            "q2": {"a": 2.0, "b": 2.0, "c": 2.0, "d": 2.0},
        }
    )

    def cfg(**kw: Any) -> RunnableConfig:
        return cast(
            RunnableConfig, {"configurable": {"services": _services(search, reranker, **kw)}}
        )

    uncut = await retriever(_state(["q1", "q2"]), cfg())
    assert [r["chunk_id"] for r in uncut["retrieved"]] == ["a", "b", "c", "d"]
    cut = await retriever(_state(["q1", "q2"]), cfg(delivery_score_cut=0.85, delivery_min_chunks=1))
    # a (1.0) and b (0.95) pass the cut; c and d do not and the floor (2 sub-queries) is already filled
    assert sorted(r["chunk_id"] for r in cut["retrieved"]) == ["a", "b"]
    floored = await retriever(
        _state(["q1", "q2"]), cfg(delivery_score_cut=0.99, delivery_min_chunks=1)
    )
    # only a passes 0.99, but the floor of one chunk per sub-query keeps b as well
    assert sorted(r["chunk_id"] for r in floored["retrieved"]) == ["a", "b"]
