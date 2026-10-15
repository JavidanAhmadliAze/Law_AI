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
    provision: str = ""


class LawChunk(BaseModel):
    chunk_id: str  # deterministic (doc + article + position) → idempotent indexing
    text: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class LegHit(BaseModel):
    """How one retrieval leg saw a chunk, before fusion collapsed the scales."""

    rank: int
    score: float
    rrf_contribution: float = 0.0


class RetrievedChunk(BaseModel):
    chunk: LawChunk
    score: float
    legs: dict[str, LegHit] = Field(default_factory=dict)
    """Per-leg rank and raw score, keyed "dense" / "bm25".

    RRF turns two incomparable scales into one number, and the collapse hides
    the distinction that matters when a provision is missed: whether both legs
    ranked it badly, or one leg found it and the other never returned it. The
    second case is structural — a chunk in one leg scores ~1/(k+r) while a
    chunk mediocre in both scores ~2/(k+r), so a strong single-leg hit loses to
    two weak dual-leg ones. Empty when the chunk did not come from a fused
    search.
    """
