"""Translator — deterministic. Seeds the supervisor with the Polish brief.

The sources are Polish, so the research runs in Polish: we translate the English
research brief once here and hand it to the supervisor as its opening message.
The writer still answers in the user's language (it reads the original question).
"""

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from law_ai.services.agents.context import last_human, services_from_config
from law_ai.services.agents.state import AgentOutputState
from law_ai.services.translation.base import Direction


async def translator(state: AgentOutputState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    brief = state.get("rewritten_query", "") or last_human(list(state["messages"]))
    polish_brief = await services.translator.translate(brief, Direction.EN_TO_PL)
    return {"supervisor_message": [HumanMessage(content=polish_brief)]}
