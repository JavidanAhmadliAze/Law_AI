"""Embedding service contract — dense vectors for the hybrid index's kNN leg.

The model must be multilingual (Polish-capable); the id comes from env
(EMBEDDING__MODEL), never from code.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for indexing."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a query (some models use a different query prefix)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector size — used when creating the OpenSearch index mapping."""
