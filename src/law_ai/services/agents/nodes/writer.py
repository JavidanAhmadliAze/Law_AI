"""Writer — deterministic, and the only streamed node.

Emits PLAIN prose via llm.stream (not llm.generate): `.astream` is what fires
LangGraph's `on_llm_new_token`, so this node's tokens reach the SSE router
(routers/ask.py, filtering on langgraph_node="writer") as they generate. A
non-streaming `ainvoke` here would emit nothing until the call completed and the
whole answer would land in one lump.

The AIMessage returned in `messages` is for state/history only — the router
ignores it, because it arrives separately via on_chain_end and would otherwise
duplicate the streamed text.
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
    parts: list[str] = []
    async for delta in services.llm.stream(
        prompt.WRITER,
        f"User question: {question}\n\nResearch notes (verbatim Polish sources):\n{notes}",
    ):
        parts.append(delta)
    report = "".join(parts)
    return {"final_report": report, "messages": [AIMessage(content=report)]}
