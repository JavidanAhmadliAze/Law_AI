import hashlib

from law_ai.acts import LegalAct
from law_ai.schemas.chunk import ChunkMetadata, LawChunk
from law_ai.services.chunking.base import RawChunk
from law_ai.services.metadata.base import BaseMetadataBuilder


class ActMetadataBuilder(BaseMetadataBuilder):
    """Act-agnostic enrichment — everything act-specific comes from the registry."""

    def build(self, raw_chunks: list[RawChunk], *, act: LegalAct) -> list[LawChunk]:
        chunks: list[LawChunk] = []
        for raw in raw_chunks:
            # deterministic id: same act+article+position+content → same _id → upsert
            digest = hashlib.sha256(
                f"{act.act_id}|{raw.article}|{raw.position}|{raw.text}".encode()
            ).hexdigest()[:16]
            chunks.append(
                LawChunk(
                    chunk_id=f"{act.act_id}-{digest}",
                    text=raw.text,
                    metadata=ChunkMetadata(
                        article=raw.article,
                        chapter=raw.chapter,
                        title=raw.text.splitlines()[0][:120] if raw.text else "",
                        act=act.act_id,
                        act_name=act.name,
                        domain=act.domain,
                        source_url=act.url,
                        language="pl",
                        effective_date=act.effective_date,
                    ),
                )
            )
        return chunks
