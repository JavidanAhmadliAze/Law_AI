"""Structured outputs for the deterministic nodes.

The guardian and query_rewriter use these with the LLM's structured-output path.
The supervisor's tools (ConductResearch/ResearchComplete/think_tool) live in
tools.py — they're bound to the model, not parsed as structured output.
"""

from pydantic import BaseModel, Field


class GuardianVerdict(BaseModel):
    allowed: bool = Field(description="True if the question may enter the pipeline")
    reason: str = Field(description="One of: ok | off_topic | injection | unsafe")
    message: str = Field(default="", description="User-facing refusal text when blocked")


class RewrittenQuery(BaseModel):
    research_brief_pl: str = Field(
        description="A single, self-contained research brief IN POLISH describing what "
        "must be researched to answer the user's question. This is what retrieval "
        "searches on, so it must use the terminology the statutes themselves use."
    )
    sub_queries_pl: list[str] = Field(
        default_factory=list,
        description="Search queries IN POLISH, one per genuinely independent legal issue. "
        "Leave EMPTY unless the question really covers separate issues that different "
        "articles govern — splitting a single-issue question only dilutes retrieval.",
    )
