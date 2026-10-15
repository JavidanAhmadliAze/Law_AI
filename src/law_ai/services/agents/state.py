import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState

from law_ai.services.agents.schema import ACTS




class AgentInputState(MessagesState):
    pass


class AgentState(MessagesState):
    analysis: dict[str, Any]  # the analyse stage's output, as recorded
    research_brief_pl: str
    search_queries_pl: dict[str, str]  # sub-query -> the ONE act it is confined to ("" = all)
    notes: Annotated[list[str], operator.add]
    retrieved: list[dict[str, Any]]
    final_report: str
    acts: list[ACTS]  # the acts that were searched, in order of first search

