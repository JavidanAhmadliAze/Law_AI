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
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from law_ai.services.agents import prompt
from law_ai.services.agents.context import services_from_config
from law_ai.services.agents.nodes.researcher import researcher_graph
from law_ai.services.agents.state import SupervisorState
from law_ai.services.agents.tools import ConductResearch, ResearchComplete, think_tool


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
    services = services_from_config(config)
    last = state["supervisor_message"][-1]
    tool_calls: list[dict[str, Any]] = list(getattr(last, "tool_calls", []) or [])
    iterations = state.get("research_iterations", 0)

    complete = (
        not tool_calls
        # .name exists at runtime (StructuredTool); mypy sees the pre-@tool class
        or any(tc["name"] == ResearchComplete.name for tc in tool_calls)  # type: ignore[attr-defined]
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

    conduct = [tc for tc in tool_calls if tc["name"] == ConductResearch.name]  # type: ignore[attr-defined]
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


def build_supervisor() -> Any:
    """The research orchestrator: supervisor ⇄ supervisor_tools (Command-routed)."""
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("supervisor_tools", supervisor_tools)
    graph.add_edge(START, "supervisor")
    return graph.compile()
