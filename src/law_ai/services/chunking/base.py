"""Chunking contract — article-aware splitting of Polish legal text."""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class RawChunk(BaseModel):
    """Chunker output before metadata enrichment."""

    text: str
    article: str = ""  # "Art. 54" when detected
    chapter: str = ""  # "Rozdział II" carried while walking the document
    position: int = 0  # order within the document (sub-chunk index included)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[RawChunk]:
        """Split document text into retrieval units aligned to articles."""
