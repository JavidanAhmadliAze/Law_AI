"""The agentic RAG pipeline (online).

    START → guardian ──blocked──────────────► END
               │ ok
           query_rewriter
               │  (Send fan-out: one sub_agent per sub-question, parallel)
           sub_agent × N ──► supervisor ──incomplete──► sub_agent × M (loop, budgeted)
                                 │ complete
                               writer ──► END

Simple questions naturally take the fast path (one sub-question → one
sub_agent); complex ones fan out and may loop once more via the supervisor.
"""

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send

from law_ai.services.agents.context import AgentServices
from law_ai.services.agents.nodes.guardian import guardian
from law_ai.services.agents.nodes.query_rewriter import query_rewriter
from law_ai.services.agents.nodes.sub_agent import sub_agent
from law_ai.services.agents.nodes.supervisor import supervisor
from law_ai.services.agents.nodes.writer import writer
from law_ai.services.agents.state import GraphState, SubAgentInput


def _route_after_guardian(state: GraphState) -> str:
    verdict = state.get("guardian_verdict")
    return "query_rewriter" if verdict is not None and verdict.allowed else END


def _dispatch_sub_agents(state: GraphState) -> list[Send]:
    return [
        Send(
            "sub_agent",
            SubAgentInput(
                sub_question=q,
                article_filter=state.get("article_filter", ""),
                query_language=state.get("query_language", "en"),
            ),
        )
        for q in state.get("sub_questions", [])
    ]


def _route_after_supervisor(state: GraphState) -> list[Send] | str:
    additional = state.get("additional_questions", [])
    if not additional:
        return "writer"
    return [
        Send(
            "sub_agent",
            SubAgentInput(
                sub_question=q,
                article_filter="",
                query_language=state.get("query_language", "en"),
            ),
        )
        for q in additional
    ]


def build_agentic_rag(services: AgentServices, checkpointer: Any = None) -> Any:
    graph: StateGraph = StateGraph(GraphState)

    # nodes are plain (state, config) coroutines; services reach them via the
    # config bound below, so no per-node closures are needed.
    graph.add_node("guardian", guardian)
    graph.add_node("query_rewriter", query_rewriter)
    # sub_agent's input is SubAgentInput (via Send), not GraphState — mypy can't
    # reconcile that with the GraphState-typed graph; correct at runtime.
    graph.add_node("sub_agent", sub_agent)  # type: ignore[arg-type]
    graph.add_node("supervisor", supervisor)
    graph.add_node("writer", writer)

    graph.add_edge(START, "guardian")
    graph.add_conditional_edges("guardian", _route_after_guardian, ["query_rewriter", END])
    graph.add_conditional_edges("query_rewriter", _dispatch_sub_agents, ["sub_agent"])
    graph.add_edge("sub_agent", "supervisor")
    graph.add_conditional_edges("supervisor", _route_after_supervisor, ["sub_agent", "writer"])
    graph.add_edge("writer", END)

    # bind services into the graph's runtime config once — every node reads it
    # via services_from_config; merges with ask.py's invoke-time config
    return graph.compile(checkpointer=checkpointer).with_config(
        {"configurable": {"services": services}}
    )
