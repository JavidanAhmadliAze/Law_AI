"""Structured schemas: typed outputs for deterministic nodes + supervisor tools.

The deterministic nodes (guardian, query_rewriter) use these with the LLM's
structured-output path; ConductResearch/ResearchComplete are bound as tools on
the supervisor model (their class names become the tool names).
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


class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""

    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be "
        "described in high detail (at least a paragraph).",
    )


class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""
