"""Reasoning tools shared by agents.

These are the ONLY tools in the graph — retrieval, translation and generation
are injected services called directly by nodes, not LLM-bound tools.

- think_tool:   self-reflection ("is the evidence sufficient?") — drives the
                sub_agent retry loop and the supervisor's go/no-go decision.
- compress_tool: context engineering — distill retrieved passages into
                {claim, source_article, quote} records (quotes verbatim, never
                translated) so GraphState never accumulates raw passage dumps.
"""

from law_ai.schemas.chunk import RetrievedChunk
from law_ai.services.agents import prompts
from law_ai.services.agents.schemas import (
    CompressedEvidence,
    EvidenceItem,
    ThinkResult,
)
from law_ai.services.llm.base import BaseLLM


async def think_tool(llm: BaseLLM, *, question: str, evidence: list[EvidenceItem]) -> ThinkResult:
    evidence_block = (
        "\n".join(f"- [{e.source_article}] {e.claim} — „{e.quote}”" for e in evidence)
        or "(no evidence gathered)"
    )
    return await llm.generate_structured(
        prompts.THINK,
        f"Question: {question}\n\nEvidence so far:\n{evidence_block}",
        ThinkResult,
    )


async def compress_tool(
    llm: BaseLLM, *, question: str, chunks: list[RetrievedChunk]
) -> list[EvidenceItem]:
    if not chunks:
        return []
    passages = "\n\n".join(
        f"[{c.chunk.metadata.article or 'unknown'}] {c.chunk.text}" for c in chunks
    )
    compressed = await llm.generate_structured(
        prompts.COMPRESS,
        f"Question: {question}\n\nRetrieved passages:\n{passages}",
        CompressedEvidence,
    )
    return compressed.items
