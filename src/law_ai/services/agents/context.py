"""Dependency injection for the agent graph.

Services + tunables are bound into the graph's runtime config once
(build_agentic_rag → .with_config); every node and tool reads them via
services_from_config — no per-node closures, no module-level singletons.
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from law_ai.services.llm.base import BaseLLM
from law_ai.services.opensearch.base import BaseSearchService


@dataclass
class AgentServices:
    llm: BaseLLM  # generation (writer)
    fast_llm: BaseLLM  # guardian / query rewriter — cheap, deterministic work
    search: BaseSearchService
    retriever_top_k: int = 12
    rerank_budget: int = 45
    min_candidates_per_query: int = 8


def services_from_config(config: RunnableConfig) -> AgentServices:
    services: Any = config["configurable"]["services"]
    return services


def last_human(messages: list[BaseMessage]) -> str:
    """The most recent user message text (the question the graph answers)."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""
