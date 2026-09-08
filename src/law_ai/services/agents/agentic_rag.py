"""General graph assembly.

Deterministic gate/rewrite/translate around the supervisor subgraph, then the
streamed writer. The two subgraphs are built where their nodes live
(nodes/researcher.py, nodes/supervisor.py); here we only wire the top level.

Services are bound into the graph's config once via .with_config, so they
propagate to every node, the supervisor subgraph, its tools, and the researcher
graph invoked inside it.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from law_ai.services.agents.context import AgentServices
from law_ai.services.agents.nodes.guardian import guardian
from law_ai.services.agents.nodes.query_rewriter import query_rewriter
from law_ai.services.agents.nodes.supervisor import build_supervisor
from law_ai.services.agents.nodes.translator import translator
from law_ai.services.agents.nodes.writer import writer
from law_ai.services.agents.state import AgentInputState, AgentOutputState


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
