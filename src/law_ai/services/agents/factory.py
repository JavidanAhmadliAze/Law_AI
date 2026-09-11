"""Agent graph factory — receives services by injection, creates none itself."""

from typing import Any

from law_ai.config import Settings
from law_ai.services.agents.agentic_rag import build_agentic_rag
from law_ai.services.agents.context import AgentServices
from law_ai.services.langfuse.client import LangfuseService
from law_ai.services.llm.base import BaseLLM
from law_ai.services.opensearch.base import BaseSearchService


def create_agent_graph(
    *,
    llm: BaseLLM,
    search: BaseSearchService,
    fast_llm: BaseLLM | None = None,
    settings: Settings,
    checkpointer: Any = None,
    langfuse: LangfuseService | None = None,
) -> Any:
    services = AgentServices(
        llm=llm,
        fast_llm=fast_llm or llm,
        search=search,
        retriever_top_k=settings.agent.retriever_top_k,
        rerank_budget=settings.agent.rerank_budget,
        min_candidates_per_query=settings.agent.min_candidates_per_query,
    )
    handler = langfuse.callback_handler() if langfuse is not None else None
    return build_agentic_rag(
        services, checkpointer=checkpointer, callbacks=[handler] if handler else None
    )
