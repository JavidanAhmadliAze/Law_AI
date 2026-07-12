"""Search service contract.

This service owns ALL retrieval mechanics: index management, hybrid search
(dense kNN + sparse BM25 with RRF fusion), metadata filtering and reranking.
Consumers (agents, Airflow tasks) call `retrieve`/`index_chunks` and never
touch OpenSearch primitives directly.
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
        """Hybrid search (+ optional metadata filters) + rerank → top_k chunks.

        `query` must be in Polish (the corpus language) — translation happens
        upstream in the agent pipeline.
        """
