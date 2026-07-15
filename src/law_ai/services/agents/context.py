"""Injected services available to agent nodes.

The agents service consumes other services — it never constructs them.
main.py's lifespan builds this container via the factories and hands it to
create_agent_graph.
"""

from dataclasses import dataclass

from law_ai.services.llm.base import BaseLLM
from law_ai.services.opensearch.base import BaseSearchService
from law_ai.services.translation.base import BaseTranslator


@dataclass(frozen=True)
class AgentServices:
    llm: BaseLLM
    search: BaseSearchService
    translator: BaseTranslator
