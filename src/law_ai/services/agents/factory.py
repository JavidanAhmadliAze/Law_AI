"""Agent graph factory — receives services by injection, creates none itself."""

from typing import Any

from law_ai.config import Settings
from law_ai.services.agents.agentic_rag import build_agentic_rag
from law_ai.services.agents.context import AgentServices
from law_ai.services.llm.base import BaseLLM
from law_ai.services.opensearch.base import BaseSearchService
from law_ai.services.translation.base import BaseTranslator


def create_agent_graph(
    *,
    llm: BaseLLM,
    search: BaseSearchService,
    translator: BaseTranslator,
    settings: Settings,
    checkpointer: Any = None,
) -> Any:
    services = AgentServices(
        llm=llm,
        search=search,
        translator=translator,
        max_research_iterations=settings.agent.max_research_iterations,
        researcher_max_tool_calls=settings.agent.researcher_max_tool_calls,
        retriever_top_k=settings.agent.retriever_top_k,
    )
    return build_agentic_rag(services, checkpointer=checkpointer)
