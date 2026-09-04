"""Graph states.

- AgentInputState  — what the router hands the graph (just messages).
- AgentOutputState — the general graph's working/output state.
- SupervisorState  — the supervisor subgraph (adds its private loop counter).
- ResearcherState  — the research subgraph (invoked per topic by the supervisor).

`supervisor_message` and `researcher_messages` use the add_messages reducer so
appends merge correctly; `notes` uses an additive reducer so parallel research
results concatenate instead of clobbering.
"""

import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class AgentInputState(MessagesState):
    pass


class AgentOutputState(MessagesState):
    rewritten_query: str
    supervisor_message: Annotated[Sequence[BaseMessage], add_messages]
    notes: Annotated[list[str], operator.add]
    final_report: str


class SupervisorState(AgentOutputState):
    research_iterations: int


class ResearcherState(TypedDict):
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_topic: str
    compressed_research: str
