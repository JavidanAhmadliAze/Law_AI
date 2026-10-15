"""Dependency injection for the agent graph.

Services + tunables are bound into the graph's runtime config once
(build_agentic_rag → .with_config); every node and tool reads them via
services_from_config — no per-node closures, no module-level singletons.
"""

from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from law_ai.services.llm.base import BaseLLM
from law_ai.services.opensearch.base import BaseSearchService
from law_ai.services.reranker.base import BaseReranker


@dataclass(kw_only=True)
class AgentServices:
    llm: Optional[BaseLLM] = None
    fast_llm: Optional[BaseLLM] = None
    reasoning_llm: Optional[BaseLLM] = None
    search: BaseSearchService
    reranker: Optional[BaseReranker] = None
    score_floor: float = 0.0
    max_chunks: int = 0
    rerank_candidates_size: int = 0
    max_searches: int = 0
    label_bank_path: str = ""
    decompose_queries: bool = True  # False: search the question itself, no planning call
    delivery_score_cut: float = 0.0  # 0 = deliver the top max_chunks uncut
    delivery_min_chunks: int = 2


def services_from_config(config: RunnableConfig) -> AgentServices:
    services: Any = config["configurable"]["services"]
    return services


def last_human(messages: list[BaseMessage]) -> str:
    """The most recent user message text (the question the graph answers)."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""
