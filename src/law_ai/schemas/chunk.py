"""Chunk contracts shared by the offline pipeline and the opensearch service."""

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Produced at ingest by services/metadata — powers online metadata filtering."""

    article: str = ""  # e.g. "Art. 54"
    chapter: str = ""  # e.g. "Rozdział II"
    title: str = ""  # article/section heading
    act: str = ""  # act slug from the registry, e.g. "kodeks-cywilny"
    act_name: str = ""  # official title, e.g. "Kodeks cywilny"
    domain: str = ""  # legal domain slug, e.g. "civil"
    source_url: str = ""
    language: str = "pl"
    effective_date: str = ""  # ISO date, if known


class LawChunk(BaseModel):
    chunk_id: str  # deterministic (doc + article + position) → idempotent indexing
    text: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class RetrievedChunk(BaseModel):
    chunk: LawChunk
    score: float
