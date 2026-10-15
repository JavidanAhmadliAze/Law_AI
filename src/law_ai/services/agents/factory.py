"""Agent graph factory — receives services by injection, creates none itself."""

from typing import Any

from law_ai.config import Settings
from law_ai.services.agents.agentic_rag import build_agentic_rag
from law_ai.services.agents.context import AgentServices
from law_ai.services.langfuse.client import LangfuseService
from law_ai.services.llm.base import BaseLLM
from law_ai.services.opensearch.base import BaseSearchService
from law_ai.services.reranker.base import BaseReranker




def create_agent_graph(
    *,
    llm: BaseLLM,
    search: BaseSearchService,
    reranker: BaseReranker,
    fast_llm: BaseLLM | None = None,
    reasoning_llm: BaseLLM | None = None,
    settings: Settings,
    checkpointer: Any = None,
    langfuse: LangfuseService | None = None,
) -> Any:
    services = AgentServices(
        llm=llm,
        fast_llm=fast_llm or llm,
        reasoning_llm=reasoning_llm or fast_llm or llm,
        search=search,
        reranker=reranker,
        score_floor=settings.agent.score_floor,
        max_chunks=settings.agent.max_chunks,
        rerank_candidates_size=settings.agent.rerank_candidates,
        max_searches=settings.agent.max_searches,
        label_bank_path=settings.agent.label_bank_path,
        decompose_queries=settings.agent.decompose_queries,
        delivery_score_cut=settings.agent.delivery_score_cut,
        delivery_min_chunks=settings.agent.delivery_min_chunks,
    )
    handler = langfuse.callback_handler() if langfuse is not None else None
    return build_agentic_rag(
        services, checkpointer=checkpointer, callbacks=[handler] if handler else None
    )
