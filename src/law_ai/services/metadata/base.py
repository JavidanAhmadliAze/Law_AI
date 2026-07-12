"""Metadata contract — enriches raw chunks into indexable LawChunks.

Runs at ingest; produces the fields online metadata filtering relies on.
"""

from abc import ABC, abstractmethod

from law_ai.acts import LegalAct
from law_ai.schemas.chunk import LawChunk
from law_ai.services.chunking.base import RawChunk


class BaseMetadataBuilder(ABC):
    @abstractmethod
    def build(self, raw_chunks: list[RawChunk], *, act: LegalAct) -> list[LawChunk]:
        """Attach act metadata + deterministic chunk_id (idempotent indexing)."""
