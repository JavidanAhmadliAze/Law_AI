"""GraphState — the shared blackboard carrying the conversation between agents.

Parallel sub_agents merge into `sub_results` via an additive reducer so
concurrent writes never clobber each other.
"""

import operator
from typing import Annotated, TypedDict

from law_ai.services.agents.schemas import (
    FinalAnswer,
    GuardianVerdict,
    SubAgentResult,
)


class GraphState(TypedDict, total=False):
    # input
    question: str
    history: list[dict[str, str]]  # prior turns [{role, content}]

    # guardian
    guardian_verdict: GuardianVerdict

    # query rewriting / routing
    sub_questions: list[str]
    article_filter: str
    query_language: str  # 'en' | 'pl'

    # sub-agent fan-out (additive reducer — parallel-safe)
    sub_results: Annotated[list[SubAgentResult], operator.add]

    # supervisor loop
    iterations: int
    additional_questions: list[str]

    # output
    final_answer: FinalAnswer


class SubAgentInput(TypedDict):
    """Payload each Send() hands to one sub_agent."""

    sub_question: str
    article_filter: str
    query_language: str
