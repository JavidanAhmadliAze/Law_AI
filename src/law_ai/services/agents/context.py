"""Injected services available to agent nodes.

The agents service consumes other services — it never constructs them.
main.py's lifespan builds this container via the factories and hands it to
create_agent_graph.
"""

from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

from law_ai.services.llm.base import BaseLLM
from law_ai.services.opensearch.base import BaseSearchService
from law_ai.services.translation.base import BaseTranslator

_SERVICES_KEY = "services"


@dataclass(frozen=True)
class AgentServices:
    llm: BaseLLM
    search: BaseSearchService
    translator: BaseTranslator


def services_from_config(config: RunnableConfig) -> AgentServices:
    """Pull the injected services out of the graph's runtime config.

    Bound once in build_agentic_rag via `.with_config(...)`; every node reads
    it here instead of closing over it, so nodes stay plain (state, config)
    functions.
    """
    services = config.get("configurable", {}).get(_SERVICES_KEY)
    if not isinstance(services, AgentServices):
        raise RuntimeError("AgentServices not bound to the graph config")
    return services
