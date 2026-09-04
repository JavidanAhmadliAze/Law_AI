"""Query rewriter — deterministic. Turns the question into a research brief."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from law_ai.services.agents import prompt
from law_ai.services.agents.context import last_human, services_from_config
from law_ai.services.agents.schema import RewrittenQuery
from law_ai.services.agents.state import AgentOutputState


async def query_rewriter(state: AgentOutputState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    question = last_human(list(state["messages"]))
    rewritten: RewrittenQuery = await services.llm.generate_structured(
        prompt.QUERY_REWRITER, question, RewrittenQuery
    )
    return {"rewritten_query": rewritten.research_brief}
