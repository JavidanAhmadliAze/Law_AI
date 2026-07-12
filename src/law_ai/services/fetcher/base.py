"""Fetcher contract — downloads source law documents (PDFs)."""

from abc import ABC, abstractmethod


class BaseFetcher(ABC):
    @abstractmethod
    async def fetch(self, url: str) -> bytes:
        """Download a document and return its raw bytes."""
