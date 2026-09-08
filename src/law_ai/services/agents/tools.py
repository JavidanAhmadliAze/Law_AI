"""Functional tools for the agents.

- think_tool: reflection scratchpad (used by supervisor and researcher).
- retrieve:   wraps the injected search service as a research tool. The service
              is read from the runtime config (services_from_config), so the tool
              stays a plain module-level object with no bound state.

Research topics reach the researcher already in Polish (the translator node seeds
the supervisor with the Polish brief), so retrieve searches the query as-is.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from law_ai.services.agents.context import services_from_config


@tool
class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""

    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be "
        "described in high detail (at least a paragraph).",
    )


@tool
class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""


@tool
def think_tool(reflection: str) -> str:
    """Record a strategic reflection about progress and what to do next."""
    return f"Reflection recorded: {reflection}"


@tool
async def retrieve(query: str, config: RunnableConfig) -> str:
    """Search Polish legal sources for passages relevant to the query."""
    services = services_from_config(config)
    chunks = await services.search.retrieve(query, top_k=services.retriever_top_k)
    if not chunks:
        return "No relevant passages found."
    return "\n\n".join(
        f"[{c.chunk.metadata.article} — {c.chunk.metadata.act}]\n{c.chunk.text}" for c in chunks
    )


tools_by_name: dict[str, Any] = {"retrieve": retrieve, "think_tool": think_tool}
