"""OpenSearch implementation: hybrid retrieval with RRF fusion + reranking.

Hybrid = two searches (dense kNN + BM25 over Polish-analyzed text) fused with
reciprocal rank fusion client-side, then an optional cross-encoder rerank.
Client-side RRF keeps the logic testable and works on any OpenSearch >= 2.x
without server-side pipeline setup; swapping to a server-side normalization
pipeline later only touches this class.
"""

import asyncio
import re
from contextlib import nullcontext
from typing import Any

from opensearchpy import AsyncOpenSearch, helpers

from law_ai.config import OpenSearchSettings
from law_ai.logging import get_logger
from law_ai.schemas.chunk import ChunkMetadata, LawChunk, LegHit, RetrievedChunk
from law_ai.services.embedding.base import BaseEmbedder
from law_ai.services.opensearch.base import BaseSearchService

logger = get_logger(__name__)

# keyword types make `term` metadata filters exact (no analysis)
_METADATA_MAPPING: dict[str, Any] = {
    "article": {"type": "keyword"},
    "chapter": {"type": "keyword"},
    "title": {"type": "text"},
    "act": {"type": "keyword"},
    "act_name": {"type": "text"},
    "domain": {"type": "keyword"},
    "source_url": {"type": "keyword"},
    "language": {"type": "keyword"},
    "effective_date": {"type": "keyword"},
}


class OpenSearchService(BaseSearchService):
    def __init__(
        self,
        settings: OpenSearchSettings,
        embedder: BaseEmbedder,
        tracer: Any = None,  # LangfuseService or None — spans are optional
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._tracer = tracer
        self._client: AsyncOpenSearch | None = None

    # ------------------------------------------------------------ lifecycle

    async def startup(self) -> None:
        auth = (self._settings.user, self._settings.password) if self._settings.user else None
        self._client = AsyncOpenSearch(
            hosts=[{"host": self._settings.host, "port": self._settings.port}],
            http_auth=auth,
            use_ssl=self._settings.use_ssl,
            verify_certs=self._settings.use_ssl,
        )
        await self.ensure_index()
        await self._warmup()
        logger.info("opensearch.startup", index=self._settings.index)

    async def _warmup(self) -> None:
        """Pre-load the lazy model paths so the first user query isn't slow.

        Non-fatal: a still-booting TEI or missing reranker weights shouldn't
        take the whole RAG stack down — the lazy paths retry on first use.
        """
        try:
            await self._embedder.embed_query("warmup")
            logger.info("opensearch.warmup.embedder_ready")
        except Exception as exc:
            logger.warning("opensearch.warmup.embedder_failed", error=str(exc))

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("opensearch.teardown")

    async def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            health = await self._client.cluster.health()
            return health["status"] in ("green", "yellow")
        except Exception:
            return False

    # ------------------------------------------------------------ indexing

    async def ensure_index(self) -> None:
        client = self._require_client()
        if await client.indices.exists(index=self._settings.index):
            await client.indices.put_mapping(
                index=self._settings.index,
                body={"properties": {"metadata": {"properties": _METADATA_MAPPING}}},
            )
            return
        await client.indices.create(
            index=self._settings.index,
            body={
                "settings": {
                    "index": {"knn": True},
                    "analysis": {"analyzer": {"default": {"type": await self._analyzer()}}},
                },
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "text": {"type": "text"},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": self._embedder.dimension,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "space_type": "cosinesimil",
                            },
                        },
                        "metadata": {"properties": _METADATA_MAPPING},
                    }
                },
            },
        )
        logger.info("opensearch.index_created", index=self._settings.index)

    async def index_chunks(self, chunks: list[LawChunk]) -> int:
        client = self._require_client()
        vectors = await self._embedder.embed_texts([c.text for c in chunks])
        actions = [
            {
                "_op_type": "index",  # upsert by _id → idempotent re-runs
                "_index": self._settings.index,
                "_id": chunk.chunk_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "vector": vector,
                "metadata": chunk.metadata.model_dump(),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        success, _ = await helpers.async_bulk(client, actions)
        await client.indices.refresh(index=self._settings.index)
        logger.info("opensearch.indexed", count=success)
        return int(success)



    # ------------------------------------------------------------ retrieval

    async def retrieve(
        self, query: str, *, top_k: int = 5, act: str | None = None
    ) -> list[RetrievedChunk]:
        """Hybrid search, then the UNION of each leg's top `union_per_leg`.

        Reranking is a separate step — call `rerank` on the results when you want
        it. `top_k` caps the union afterwards; the union itself is at most
        2 x union_per_leg and smaller wherever the legs agree.
        """
        span_cm = (
            self._tracer.span("retrieval", input={"query": query, "top_k": top_k, "acts": act})
            if self._tracer is not None
            else nullcontext()
        )
        with span_cm as span:
            client = self._require_client()
            query_vector = await self._embedder.embed_query(query)

            depth = self._settings.candidates_per_leg
            # metadata.act is keyword-mapped, so this is an exact term filter —
            # resolved by the index, no analysis, no scoring cost
            act_filter: list[dict[str, Any]] = [{"terms": {"metadata.act": [act]}}] if act else []

            legs_on = self._settings.legs  # hybrid | dense | bm25

            async def _empty() -> dict[str, Any]:
                return {"hits": {"hits": []}}

            knn: dict[str, Any] = {"vector": query_vector, "k": depth}
            if act_filter:
                # INSIDE the knn clause. A filter applied outside it post-filters
                # the k nearest neighbours instead of searching within the scope,
                # so a scoped search silently returns a fraction of `k`.
                knn["filter"] = {"bool": {"filter": act_filter}}
            dense_task = (
                client.search(
                    index=self._settings.index,
                    body={"size": depth, "query": {"knn": {"vector": knn}}},
                )
                if legs_on in ("hybrid", "dense")
                else _empty()
            )

            sparse_query: dict[str, Any] = self._sparse_query(query)
            if act_filter:
                sparse_query = {"bool": {"must": [sparse_query], "filter": act_filter}}
            sparse_task = (
                client.search(
                    index=self._settings.index,
                    body={"size": depth, "query": sparse_query},
                )
                if legs_on in ("hybrid", "bm25")
                else _empty()
            )
            dense_res, sparse_res = await asyncio.gather(dense_task, sparse_task)

            dense_hits, sparse_hits = self._hits(dense_res), self._hits(sparse_res)
            legs_in = {"dense": dense_hits, "bm25": sparse_hits}
            if legs_on != "hybrid":
                legs_in = {legs_on: legs_in[legs_on]}
            if self._settings.fusion == "rrf":
                fused, legs_meta = self._rrf(legs_in, depth, self._settings.rrf_k)
            else:
                fused, legs_meta = self._union(legs_in, self._settings.union_per_leg)
            results = fused[:top_k]
            # The per-leg numbers were already computed for tracing; carrying
            # them on the chunk lets a report say WHICH leg found a provision
            # instead of only that fusion ranked it somewhere.
            for r in results:
                r.legs = {
                    name: LegHit(**meta) for name, meta in legs_meta[r.chunk.chunk_id].items()
                }

            if span is not None:
                span.update(
                    output={
                        "candidates_after_union": len(fused),
                        "dense_returned": len(dense_hits),
                        "bm25_returned": len(sparse_hits),
                        # how much the two legs actually agreed. Under a union this
                        # sets the candidate count: no overlap gives 2 x per_leg,
                        # complete overlap gives per_leg.
                        "legs_overlap": len(
                            {h["_id"] for h in dense_hits} & {h["_id"] for h in sparse_hits}
                        ),
                        "results": [
                            {
                                "chunk_id": r.chunk.chunk_id,
                                "article": r.chunk.metadata.article,
                                "act": r.chunk.metadata.act,
                                "rrf_rank": rank,
                                "rrf_score": round(r.score, 5),
                                "dense_rank": legs_meta[r.chunk.chunk_id]
                                .get("dense", {})
                                .get("rank"),
                                "dense_score": legs_meta[r.chunk.chunk_id]
                                .get("dense", {})
                                .get("score"),
                                "bm25_rank": legs_meta[r.chunk.chunk_id]
                                .get("bm25", {})
                                .get("rank"),
                                "bm25_score": legs_meta[r.chunk.chunk_id]
                                .get("bm25", {})
                                .get("score"),
                                # the diagnosis in one field: which legs found it
                                "found_by": "+".join(sorted(legs_meta[r.chunk.chunk_id])),
                                "text_preview": r.chunk.text[:200],
                            }
                            for rank, r in enumerate(results, 1)
                        ],
                    }
                )
            return results

    # ------------------------------------------------------------ internals

    def _require_client(self) -> AsyncOpenSearch:
        if self._client is None:
            raise RuntimeError("OpenSearchService not started — call startup() first")
        return self._client

    def _sparse_query(self, query: str) -> dict[str, Any]:
        """The keyword leg: the provision's own wording, optionally plus its act
        title.

        At boost 0 this is a plain match on `text` rather than a multi_match with
        a zeroed field — an explicit off, not a reliance on what Lucene does with
        a weight of zero.

        When on, most_fields rather than best_fields: the title is meant to ADD a
        little to a chunk already matching on its text, not to compete with it.
        best_fields takes the max per chunk, so at any sane boost the title would
        never register at all.
        """
        boost = self._settings.act_name_boost
        if boost <= 0:
            return {"match": {"text": query}}
        return {
            "multi_match": {
                "query": query,
                "type": "most_fields",
                "fields": ["text", f"metadata.act_name^{boost}"],
            }
        }

    @staticmethod
    def _hits(response: dict[str, Any]) -> list[dict[str, Any]]:
        return response["hits"]["hits"]

    @staticmethod
    def _rrf(
        legs: dict[str, list[dict[str, Any]]], per_leg: int, k: int
    ) -> tuple[list[RetrievedChunk], dict[str, dict[str, Any]]]:
        """Reciprocal-rank fusion: score = sum over legs of 1 / (k + rank).

        Kept beside `_union` for measurement only — see `_union` for why the
        production pool is a union. The caller cuts the fused list to `top_k`,
        which is where the two differ: a union never drops a leg's top hit, RRF
        can rank it below two chunks mediocre in both legs.
        """
        scores: dict[str, float] = {}
        sources: dict[str, dict[str, Any]] = {}
        legs_meta: dict[str, dict[str, Any]] = {}
        for leg_name, leg in legs.items():
            for rank, hit in enumerate(leg[:per_leg], 1):
                chunk_id = hit["_id"]
                sources[chunk_id] = hit["_source"]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
                legs_meta.setdefault(chunk_id, {})[leg_name] = {
                    "rank": rank,
                    "score": round(float(hit.get("_score") or 0.0), 4),
                }
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        return [
            RetrievedChunk(
                chunk=LawChunk(
                    chunk_id=chunk_id,
                    text=sources[chunk_id]["text"],
                    metadata=ChunkMetadata(**sources[chunk_id].get("metadata", {})),
                ),
                score=round(score, 5),
            )
            for chunk_id, score in ranked
        ], legs_meta

    @staticmethod
    def _union(
        legs: dict[str, list[dict[str, Any]]], per_leg: int
    ) -> tuple[list[RetrievedChunk], dict[str, dict[str, Any]]]:
        """Each leg's top `per_leg`, merged — no fusion score.

        Replaces reciprocal rank fusion, which decided the candidate set by a
        number that hid what a report needs. RRF gives a chunk present in one leg
        ~1/(k+r) and a chunk mediocre in both ~2/(k+r), so a strong single-leg hit
        loses to two weak dual-leg hits however strong it is: measured, a provision
        at dense rank 1 sat at fused rank 16 because BM25 never returned it.

        A union has no such interaction. Each leg contributes its own best
        `per_leg` and the cross-encoder decides among them, which is the step that
        can actually read the provision. The cost is that the candidate set is no
        longer ordered by anything meaningful before reranking — `score` here is
        the reciprocal of the best rank the chunk achieved in either leg, enough
        to make the order deterministic and nothing more.

        A chunk both legs returned appears once, so the union is at most
        2 x per_leg and usually less.
        """
        best_rank: dict[str, int] = {}
        sources: dict[str, dict[str, Any]] = {}
        legs_meta: dict[str, dict[str, Any]] = {}
        for leg_name, leg in legs.items():
            for rank, hit in enumerate(leg[:per_leg]):
                chunk_id = hit["_id"]
                sources[chunk_id] = hit["_source"]
                best_rank[chunk_id] = min(best_rank.get(chunk_id, rank + 1), rank + 1)
                legs_meta.setdefault(chunk_id, {})[leg_name] = {
                    "rank": rank + 1,
                    "score": round(float(hit.get("_score") or 0.0), 4),
                }
        ranked = sorted(best_rank.items(), key=lambda item: item[1])
        return [
            RetrievedChunk(
                chunk=LawChunk(
                    chunk_id=chunk_id,
                    text=sources[chunk_id]["text"],
                    metadata=ChunkMetadata(**sources[chunk_id].get("metadata", {})),
                ),
                score=round(1.0 / rank, 5),
            )
            for chunk_id, rank in ranked
        ], legs_meta


    async def get_chunks(
        self,
        *,
        article: str | None = None,
        act: str | None = None,
        size: int = 1000,
    ) -> list[dict]:
        """Retrieve chunks directly by metadata, without vector/BM25 search."""

        client = self._require_client()

        filters = []

        if article:
            filters.append({"term": {"metadata.article": article}})

        if act:
            filters.append({"term": {"metadata.act": act}})

        query = (
            {"bool": {"filter": filters}}
            if filters
            else {"match_all": {}}
        )

        response = await client.search(
            index=self._settings.index,
            body={
                "size": size,
                "_source": [
                    "text",
                    "metadata",
                ],
                "query": query,
            },
        )

        return [
            {
                "chunk_id": hit["_id"],
                **hit["_source"],
            }
            for hit in response["hits"]["hits"]
        ]
    async def update_chunk_metadata(
        self,
        chunk_id: str,
        metadata: dict[str, Any],
    ) -> None:
        client = self._require_client()

        await client.update(
            index=self._settings.index,
            id=chunk_id,
            body={
                "script": {
                    "lang": "painless",
                    "source": """
                        for (entry in params.metadata.entrySet()) {
                            ctx._source.metadata[entry.getKey()] = entry.getValue();
                        }

                        ctx._source.metadata.remove('institution');

                    """,
                    "params": {
                        "metadata": metadata,
                    },
                }
            },
        )
    async def count_chunks(
    self,
    *,
    act: str | None = None,
    article: str | None = None,
) -> int:

        client = self._require_client()

        filters = []

        if act:
            filters.append({"term": {"metadata.act": act}})

        if article:
            filters.append({"term": {"metadata.article": article}})

        query = (
            {"bool": {"filter": filters}}
            if filters
            else {"match_all": {}}
        )

        response = await client.count(
            index=self._settings.index,
            body={"query": query},
        )

        return response["count"]


_ARTICLE_NUMBER = re.compile(r"(\d+)([a-z\u0105\u0107\u0119\u0142\u0144\xf3\u015b\u017a\u017c]*)")


def _article_key(article: str) -> tuple[int, str]:
    """Sort "Art. 116a" after "Art. 116" and both before "Art. 117"."""
    match = _ARTICLE_NUMBER.search(article)
    return (int(match.group(1)), match.group(2)) if match else (10**9, article)


def _first_key(agg: dict[str, Any]) -> str:
    buckets = agg.get("buckets") or []
    return str(buckets[0]["key"]) if buckets else ""
