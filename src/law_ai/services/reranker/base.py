"""Reranking contract.

Separate from the search service on purpose. A vector store's job is storage and
similarity — kNN, BM25, fusion, metadata filters. Reranking is a second-stage
model that reads (query, passage) pairs and scores them jointly; it touches no
index and would have to be reimplemented verbatim if the store were swapped.

Keeping it its own port also means the CPU, GPU and remote-service versions are
sibling adapters rather than edits to the store.
"""

from abc import ABC, abstractmethod

from law_ai.schemas.chunk import RetrievedChunk


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Score `candidates` against `query` and return them best-first.

        Scores are query-conditional — the same passage scores differently
        against different queries — so `query` must be the one these candidates
        should be judged against, and scores from separate calls are not
        comparable with each other.

        Implementations must be a no-op (returning the input order) when no
        model is configured, so reranking can be switched off by configuration
        alone.
        """
