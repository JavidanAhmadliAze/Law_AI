"""Guardian — the entry gate. Deterministic: one structured LLM call.

Routes with a Command: on to planning when allowed, straight to END with a
refusal message when not (no verdict field needed in the state).
"""

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from law_ai.services.agents import prompt
from law_ai.services.agents.context import last_human, services_from_config
from law_ai.services.agents.schema import GuardianVerdict
from law_ai.services.agents.state import AgentState


async def guardian(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["query_rewriter", "__end__"]]:
    services = services_from_config(config)
    question = last_human(list(state["messages"]))
    llm = services.fast_llm or services.llm
    if llm is None:
        raise RuntimeError("guardian needs an LLM in the agent services")
    verdict: GuardianVerdict = await llm.generate_structured(
        prompt.GUARDIAN, question, GuardianVerdict
    )
    if not verdict.allowed:
        message = verdict.message or "I can only help with Polish legal questions."
        update: dict[str, Any] = {
            "final_report": message,
            "messages": [AIMessage(content=message)],
        }
        return Command(goto=END, update=update)  # type: ignore[arg-type]
    return Command(goto="query_rewriter")
