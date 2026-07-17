"""Cache contract — exact-match memoization of full RAG answers.

A hit skips the entire agent pipeline (translation, retrieval, reranking,
generation), so identical questions answer in milliseconds instead of
minutes. Keys incorporate the pipeline fingerprint (models, index): swapping
any model invalidates old entries automatically.
"""

from abc import ABC, abstractmethod

from law_ai.schemas.chat import AskRequest, AskResponse


class BaseCache(ABC):
    @abstractmethod
    async def startup(self) -> None:
        """Open/verify the backend connection (ping)."""

    @abstractmethod
    async def teardown(self) -> None:
        """Close the backend connection."""

    @abstractmethod
    async def find_cached_response(self, request: AskRequest) -> AskResponse | None:
        """Return the cached answer for an identical request, or None."""

    @abstractmethod
    async def store_response(self, request: AskRequest, response: AskResponse) -> None:
        """Store an answer under the request's key with the configured TTL."""
