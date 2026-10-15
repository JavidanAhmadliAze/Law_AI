"""Graph assembly — a straight chain, no loops.

    guardian → planning → retriever → writer

The only loop in the graph is inside the planning subgraph, where a stage that
broke a checkable rule retries itself once. Everything else runs exactly once,
so a question costs four LLM calls (guardian, analyse, plan, writer) plus at
most two retries, and one concurrent burst of searches. The only branch out here
is the guardian's refusal, which jumps to END.

Breadth that an agent loop would have bought by searching repeatedly is bought
instead by the planner emitting several sub-queries up front — decided once,
from the question alone, and executed in parallel. Planning is split in two
because deciding the institution and wording the searches in one call let a
misread institution propagate into every sub-query unchecked.

Services (and the tracing callback, when enabled) are bound into the graph's
config once via .with_config, so they propagate to every node.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from law_ai.services.agents.context import AgentServices
from law_ai.services.agents.nodes.guardian import guardian
from law_ai.services.agents.nodes.query_rewriter import query_rewriter
from law_ai.services.agents.nodes.retriever import retriever
from law_ai.services.agents.nodes.writer import writer
from law_ai.services.agents.state import AgentInputState, AgentState


def build_agentic_rag(
    services: AgentServices,
    checkpointer: Any = None,
    callbacks: list[Any] | None = None,
) -> Any:
    graph = StateGraph(AgentState, input_schema=AgentInputState)
    graph.add_node("guardian", guardian)
    # A compiled subgraph is a node: its internal retries are its own business,
    # and only the keys its `finalize` returns reach this state.
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("retriever", retriever)
    graph.add_node("writer", writer)

    graph.add_edge(START, "guardian")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "writer")
    graph.add_edge("writer", END)

    config: RunnableConfig = {"configurable": {"services": services}}
    if callbacks:
        # bound here rather than per-request so offline callers (eval runs,
        # scripts) are traced too. Callers add per-request metadata only —
        # supplying callbacks again would attach the handler twice.
        config["callbacks"] = callbacks
    return graph.compile(checkpointer=checkpointer).with_config(config)
