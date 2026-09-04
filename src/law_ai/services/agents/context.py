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
from law_ai.services.translation.base import BaseTranslator


@dataclass
class AgentServices:
    llm: BaseLLM
    search: BaseSearchService
    translator: BaseTranslator
    max_research_iterations: int = 3
    researcher_max_tool_calls: int = 5
    retriever_top_k: int = 5


def services_from_config(config: RunnableConfig) -> AgentServices:
    services: Any = config["configurable"]["services"]
    return services


def last_human(messages: list[BaseMessage]) -> str:
    """The most recent user message text (the question the graph answers)."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""
