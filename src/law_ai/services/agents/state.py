"""Graph states.

- AgentInputState  — what the router hands the graph (just messages).
- AgentOutputState — the chain's working/output state.

The chain is linear, so no channel needs a merge reducer for concurrent writers;
`notes` keeps an additive reducer only because `messages` semantics come from
MessagesState and notes are appended alongside them.
"""

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState


class AgentInputState(MessagesState):
    pass


class AgentOutputState(MessagesState):
    research_brief_pl: str  # the rewriter's reading of the question (Polish); traced, not searched
    # What retrieval actually runs — ALWAYS at least one. The rewriter decides
    # how many searches a question needs; the retriever only decides how to
    # spend its budget across them.
    search_queries_pl: list[str]
    # optional metadata narrowing for retrieval, e.g. {"domain": "employment"}
    metadata_filters: dict[str, str]
    notes: Annotated[list[str], operator.add]
    # what the notes were built from: article, act and cross-encoder score. The
    # formatted notes lose all of it, and it is what tells you whether a passage
    # won on merit or was carried in by its article.
    retrieved: list[dict[str, Any]]
    final_report: str
