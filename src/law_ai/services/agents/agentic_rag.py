"""Graph assembly.

Three StateGraphs:
- researcher_graph — the research sub-agent (llm_call ↔ tool_node → compress).
  Compiled once at module load; supervisor_tools invokes it (in parallel).
- supervisor_graph — the non-deterministic orchestrator (supervisor ↔ tools),
  embedded as a subgraph node in the general graph.
- the general graph — deterministic gate/rewrite/translate around the supervisor
  subgraph, then the streamed writer.

Services are bound into the general graph's config once via .with_config, so they
propagate to every node, subgraph, tool, and the researcher graph invoked inside.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from law_ai.services.agents.context import AgentServices
from law_ai.services.agents.nodes.guardian import guardian
from law_ai.services.agents.nodes.query_rewriter import query_rewriter
from law_ai.services.agents.nodes.researcher import (
    compress_research,
    llm_call,
    should_continue,
    tool_node,
)
from law_ai.services.agents.nodes.supervisor import supervisor, supervisor_tools
from law_ai.services.agents.nodes.translator import translator
from law_ai.services.agents.nodes.writer import writer
from law_ai.services.agents.state import (
    AgentInputState,
    AgentOutputState,
    ResearcherState,
    SupervisorState,
)


def build_researcher() -> Any:
    graph = StateGraph(ResearcherState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("compress_research", compress_research)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges(
        "llm_call",
        should_continue,
        {"tool_node": "tool_node", "compress_research": "compress_research"},
    )
    graph.add_edge("tool_node", "llm_call")
    graph.add_edge("compress_research", END)
    return graph.compile()


# compiled once; invoked (in parallel) inside supervisor_tools
researcher_graph = build_researcher()


def build_supervisor() -> Any:
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("supervisor_tools", supervisor_tools)
    graph.add_edge(START, "supervisor")
    # supervisor and supervisor_tools route via Command(goto=...)
    return graph.compile()


def build_agentic_rag(services: AgentServices, checkpointer: Any = None) -> Any:
    graph = StateGraph(AgentOutputState, input_schema=AgentInputState)
    graph.add_node("guardian", guardian)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("translator", translator)
    graph.add_node("supervisor", build_supervisor())  # subgraph
    graph.add_node("writer", writer)

    graph.add_edge(START, "guardian")
    # guardian routes via Command → query_rewriter or END
    graph.add_edge("query_rewriter", "translator")
    graph.add_edge("translator", "supervisor")
    graph.add_edge("supervisor", "writer")
    graph.add_edge("writer", END)

    return graph.compile(checkpointer=checkpointer).with_config(
        {"configurable": {"services": services}}
    )
