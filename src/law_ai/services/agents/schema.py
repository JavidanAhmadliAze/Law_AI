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
    research_brief: str = Field(
        description="A single, self-contained research brief in English describing what "
        "must be researched in Polish law to answer the user's question."
    )
