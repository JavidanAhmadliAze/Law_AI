"""Search service contract.

This service owns ALL retrieval mechanics: index management, hybrid search
(dense kNN + sparse BM25 with RRF fusion), metadata filtering and reranking.
Consumers (agents, Airflow tasks) call `retrieve`/`rerank`/`index_chunks` and
never touch OpenSearch primitives directly.

`retrieve` and `rerank` are separate operations on purpose: retrieval is
query-scoped and cheap, reranking is expensive and query-conditional, so the
caller chooses the pool to score rather than paying for a rerank on every
search.
"""

from abc import ABC, abstractmethod

from law_ai.schemas.chunk import LawChunk, RetrievedChunk


class BaseSearchService(ABC):
    @abstractmethod
    async def startup(self) -> None:
        """Open the client and ensure the index exists."""

    @abstractmethod
    async def teardown(self) -> None:
        """Close the client."""

    @abstractmethod
    async def health_check(self) -> bool: ...

    @abstractmethod
    async def index_chunks(self, chunks: list[LawChunk]) -> int:
        """Embed + upsert chunks (idempotent via chunk_id). Returns count."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid search (dense kNN + BM25, RRF-fused) → top_k chunks.

        No reranking — call `rerank` on the result when you want it.

        `query` must be in Polish (the corpus language) — translation happens
        upstream in the agent pipeline.
        """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Cross-encoder rerank of a candidate set against `query`.

        Accepts candidates from any number of `retrieve` calls, so a caller can
        pool them and score once. No-op when no reranker is configured.
        """
