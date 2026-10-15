"""Search service contract.

This service owns ALL retrieval mechanics: index management, hybrid search
(dense kNN + sparse BM25 with RRF fusion), metadata filtering and reranking.
Consumers (agents, Airflow tasks) call `retrieve`/`index_chunks` and never
touch OpenSearch primitives directly. Reranking is a separate port
(services/reranker) — it scores pairs with a model and touches no index.

Retrieval is query-scoped and cheap; reranking is expensive, model-based and
lives behind its own interface, so a caller chooses what to score rather than
paying for it on every search.
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
        self, query: str, *, top_k: int = 5, act: str | None = None
    ) -> list[RetrievedChunk]:
        """Hybrid search (dense kNN ∪ BM25) → up to top_k candidate chunks.

        No reranking — call `rerank` on the result when you want it.

        `query` must be in Polish (the corpus language).

        `act` confines BOTH legs to that act_id before scoring, so provisions
        outside it never compete. None searches everything.

        """
