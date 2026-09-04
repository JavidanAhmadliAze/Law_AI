"""Writer — deterministic, and the streamed node.

Emits PLAIN prose via llm.generate so LangGraph's stream_mode="messages" streams
this node's tokens to the router (the router filters on langgraph_node="writer").
Grounds the answer only in the collected research notes.
"""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from law_ai.services.agents import prompt
from law_ai.services.agents.context import last_human, services_from_config
from law_ai.services.agents.state import AgentOutputState


async def writer(state: AgentOutputState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    question = last_human(list(state["messages"]))
    notes = "\n\n".join(state.get("notes", [])) or "(no research notes were gathered)"
    report = await services.llm.generate(
        prompt.WRITER,
        f"User question: {question}\n\nResearch notes (verbatim Polish sources):\n{notes}",
    )
    return {"final_report": report, "messages": [AIMessage(content=report)]}
