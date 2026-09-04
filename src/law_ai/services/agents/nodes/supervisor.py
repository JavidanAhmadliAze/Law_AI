"""Supervisor nodes — the non-deterministic research orchestrator.

- supervisor:       LLM with tools (ConductResearch, ResearchComplete, think_tool)
                    decides what to research next. Routes to supervisor_tools.
- supervisor_tools: executes the tool calls — think reflections inline, and
                    ConductResearch delegations by invoking the research subgraph
                    (in parallel via asyncio.gather). Routes back to supervisor,
                    or to END when research is complete / capped.
"""

import asyncio
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from law_ai.services.agents import prompt
from law_ai.services.agents.context import services_from_config
from law_ai.services.agents.schema import ConductResearch, ResearchComplete
from law_ai.services.agents.state import SupervisorState
from law_ai.services.agents.tools import think_tool


async def supervisor(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    services = services_from_config(config)
    model = services.llm.chat_model.bind_tools([ConductResearch, ResearchComplete, think_tool])
    messages = [SystemMessage(content=prompt.SUPERVISOR), *state["supervisor_message"]]
    response = await model.ainvoke(messages)
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_message": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    # lazy import: the compiled research graph lives in agentic_rag (avoids a cycle)
    from law_ai.services.agents.agentic_rag import researcher_graph

    services = services_from_config(config)
    last = state["supervisor_message"][-1]
    tool_calls: list[dict[str, Any]] = list(getattr(last, "tool_calls", []) or [])
    iterations = state.get("research_iterations", 0)

    complete = (
        not tool_calls
        or any(tc["name"] == ResearchComplete.__name__ for tc in tool_calls)
        or iterations >= services.max_research_iterations
    )
    if complete:
        return Command(goto=END)  # type: ignore[arg-type]

    tool_messages: list[ToolMessage] = []
    notes: list[str] = []

    for tc in tool_calls:
        if tc["name"] == think_tool.name:
            observation = think_tool.invoke(tc["args"])
            tool_messages.append(
                ToolMessage(content=str(observation), name=tc["name"], tool_call_id=tc["id"])
            )

    conduct = [tc for tc in tool_calls if tc["name"] == ConductResearch.__name__]
    if conduct:
        results = await asyncio.gather(
            *[
                researcher_graph.ainvoke(
                    {
                        "researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])],
                        "research_topic": tc["args"]["research_topic"],
                    },
                    config,
                )
                for tc in conduct
            ]
        )
        for tc, result in zip(conduct, results, strict=True):
            note = result.get("compressed_research", "")
            notes.append(note)
            tool_messages.append(ToolMessage(content=note, name=tc["name"], tool_call_id=tc["id"]))

    return Command(goto="supervisor", update={"supervisor_message": tool_messages, "notes": notes})
