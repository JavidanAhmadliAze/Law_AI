"""OpenSearch implementation: hybrid retrieval with RRF fusion + reranking.

Hybrid = two searches (dense kNN + BM25 over Polish-analyzed text) fused with
reciprocal rank fusion client-side, then an optional cross-encoder rerank.
Client-side RRF keeps the logic testable and works on any OpenSearch >= 2.x
without server-side pipeline setup; swapping to a server-side normalization
pipeline later only touches this class.
"""

import asyncio
from contextlib import nullcontext
from typing import Any

from opensearchpy import AsyncOpenSearch, helpers

from law_ai.config import OpenSearchSettings, RerankerSettings
from law_ai.logging import get_logger
from law_ai.schemas.chunk import ChunkMetadata, LawChunk, RetrievedChunk
from law_ai.services.embedding.base import BaseEmbedder
from law_ai.services.opensearch.base import BaseSearchService

logger = get_logger(__name__)

_RRF_K = 60  # standard reciprocal-rank-fusion constant
_CANDIDATES_PER_LEG = 25  # candidates fetched per leg before fusion/rerank

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
        reranker_settings: RerankerSettings,
        embedder: BaseEmbedder,
        tracer: Any = None,  # LangfuseService or None — spans are optional
    ) -> None:
        self._settings = settings
        self._reranker_settings = reranker_settings
        self._embedder = embedder
        self._tracer = tracer
        self._client: AsyncOpenSearch | None = None
        self._reranker: Any = None  # lazy cross-encoder

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
        if self._reranker_settings.model:
            try:
                reranker = await self._ensure_reranker()
                await asyncio.to_thread(reranker.predict, [("warmup", "warmup")])
                logger.info("opensearch.warmup.reranker_ready")
            except Exception as exc:
                logger.warning("opensearch.warmup.reranker_failed", error=str(exc))

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
            # additive + idempotent: picks up metadata fields added after the
            # index was first created (e.g. act/domain), keeping them keyword
            # so term filters stay exact instead of dynamic text mappings
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
                    "analysis": {
                        # Polish morphology matters for BM25; the stempel
                        # plugin ("polish" analyzer) is ideal — fall back is
                        # handled at query time, standard analyzer still works.
                        "analyzer": {"default": {"type": "standard"}}
                    },
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
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        span_cm = (
            self._tracer.span(
                "retrieval", input={"query": query, "top_k": top_k, "filters": filters}
            )
            if self._tracer is not None
            else nullcontext()
        )
        with span_cm as span:
            client = self._require_client()
            filter_clause = self._build_filters(filters)
            query_vector = await self._embedder.embed_query(query)

            dense_task = client.search(
                index=self._settings.index,
                body={
                    "size": _CANDIDATES_PER_LEG,
                    "query": self._wrap_filter(
                        {"knn": {"vector": {"vector": query_vector, "k": _CANDIDATES_PER_LEG}}},
                        filter_clause,
                    ),
                },
            )
            sparse_task = client.search(
                index=self._settings.index,
                body={
                    "size": _CANDIDATES_PER_LEG,
                    "query": self._wrap_filter({"match": {"text": query}}, filter_clause),
                },
            )
            dense_res, sparse_res = await asyncio.gather(dense_task, sparse_task)

            fused = self._rrf_fuse([self._hits(dense_res), self._hits(sparse_res)])
            candidates = fused[: max(top_k * 3, top_k)]  # rerank pool
            reranked = await self._rerank(query, candidates)
            results = reranked[:top_k]

            if span is not None:
                rrf_by_id = {c.chunk.chunk_id: c.score for c in candidates}
                span.update(
                    output={
                        "rerank_enabled": bool(self._reranker_settings.model),
                        "candidates_after_fusion": len(fused),
                        "rerank_pool": len(candidates),
                        "results": [
                            {
                                "chunk_id": r.chunk.chunk_id,
                                "article": r.chunk.metadata.article,
                                "act": r.chunk.metadata.act,
                                "rrf_score": round(rrf_by_id.get(r.chunk.chunk_id, 0.0), 5),
                                "rerank_score": round(r.score, 5),
                                "text_preview": r.chunk.text[:200],
                            }
                            for r in results
                        ],
                    }
                )
            return results

    # ------------------------------------------------------------ internals

    def _require_client(self) -> AsyncOpenSearch:
        if self._client is None:
            raise RuntimeError("OpenSearchService not started — call startup() first")
        return self._client

    @staticmethod
    def _build_filters(filters: dict[str, str] | None) -> list[dict[str, Any]]:
        if not filters:
            return []
        return [{"term": {f"metadata.{key}": value}} for key, value in filters.items() if value]

    @staticmethod
    def _wrap_filter(query: dict[str, Any], filter_clause: list[dict[str, Any]]) -> dict[str, Any]:
        if not filter_clause:
            return query
        return {"bool": {"must": [query], "filter": filter_clause}}

    @staticmethod
    def _hits(response: dict[str, Any]) -> list[dict[str, Any]]:
        return response["hits"]["hits"]

    @staticmethod
    def _rrf_fuse(legs: list[list[dict[str, Any]]]) -> list[RetrievedChunk]:
        """Reciprocal rank fusion: score = Σ 1/(k + rank) across legs."""
        scores: dict[str, float] = {}
        sources: dict[str, dict[str, Any]] = {}
        for leg in legs:
            for rank, hit in enumerate(leg):
                chunk_id = hit["_id"]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
                sources[chunk_id] = hit["_source"]
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            RetrievedChunk(
                chunk=LawChunk(
                    chunk_id=chunk_id,
                    text=sources[chunk_id]["text"],
                    metadata=ChunkMetadata(**sources[chunk_id].get("metadata", {})),
                ),
                score=score,
            )
            for chunk_id, score in ranked
        ]

    async def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Cross-encoder rerank; no-op when RERANKER__MODEL is unset."""
        if not self._reranker_settings.model or not candidates:
            return candidates
        reranker = await self._ensure_reranker()
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = await asyncio.to_thread(reranker.predict, pairs)
        reranked = sorted(
            zip(candidates, scores, strict=True), key=lambda item: item[1], reverse=True
        )
        return [RetrievedChunk(chunk=c.chunk, score=float(s)) for c, s in reranked]

    async def _ensure_reranker(self) -> Any:
        if self._reranker is None:

            def _load() -> Any:
                from sentence_transformers import CrossEncoder  # lazy: heavy import

                return CrossEncoder(self._reranker_settings.model)

            self._reranker = await asyncio.to_thread(_load)
        return self._reranker
