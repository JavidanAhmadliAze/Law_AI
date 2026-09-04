"""Researcher nodes — the non-deterministic research sub-agent.

- llm_call:          decide whether to call tools (retrieve/think) or answer.
                     Tools are unbound once the per-agent call cap is hit, which
                     forces a final synthesis and ends the loop.
- tool_node:         execute the model's tool calls and return their observations.
- should_continue:   loop to tool_node while there are tool calls, else compress.
- compress_research: distill the raw findings into notes, keeping every quote.
"""

from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from law_ai.services.agents import prompt
from law_ai.services.agents.context import services_from_config
from law_ai.services.agents.state import ResearcherState
from law_ai.services.agents.tools import retrieve, think_tool, tools_by_name


async def llm_call(state: ResearcherState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    tool_turns = sum(
        1
        for m in state["researcher_messages"]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )
    messages = [SystemMessage(content=prompt.RESEARCHER), *state["researcher_messages"]]
    if tool_turns >= services.researcher_max_tool_calls:
        model: Any = services.llm.chat_model  # cap reached → no tools → force synthesis
    else:
        model = services.llm.chat_model.bind_tools([retrieve, think_tool])
    response = await model.ainvoke(messages)
    return {"researcher_messages": [response]}


async def tool_node(state: ResearcherState, config: RunnableConfig) -> dict[str, Any]:
    tool_calls = getattr(state["researcher_messages"][-1], "tool_calls", []) or []
    outputs: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = await tool.ainvoke(tool_call["args"], config)
        outputs.append(
            ToolMessage(
                content=str(observation), name=tool_call["name"], tool_call_id=tool_call["id"]
            )
        )
    return {"researcher_messages": outputs}


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    last = state["researcher_messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_node"
    return "compress_research"


async def compress_research(state: ResearcherState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    findings = (
        "\n\n".join(
            str(m.content) for m in state["researcher_messages"] if isinstance(m, ToolMessage)
        )
        or "(no findings were retrieved)"
    )
    compressed = await services.llm.generate(
        prompt.COMPRESS,
        f"Research topic: {state['research_topic']}\n\nRaw findings:\n{findings}",
    )
    return {"compressed_research": compressed}
