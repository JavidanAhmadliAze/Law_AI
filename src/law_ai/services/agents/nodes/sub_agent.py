"""Sub-agent — answers ONE sub-question autonomously.

translate (service) → retrieve (service) → compress_tool → think_tool →
optionally one refined retrieval round. Returns compact evidence only;
raw passages never reach GraphState.
"""

from collections.abc import Awaitable, Callable

from law_ai.logging import get_logger
from law_ai.services.agents.context import AgentServices
from law_ai.services.agents.schemas import SubAgentResult
from law_ai.services.agents.state import SubAgentInput
from law_ai.services.agents.tools import compress_tool, think_tool
from law_ai.services.translation.base import Direction

logger = get_logger(__name__)

_TOP_K = 5


def build_sub_agent(services: AgentServices) -> Callable[[SubAgentInput], Awaitable[dict]]:
    async def sub_agent(payload: SubAgentInput) -> dict:
        sub_question = payload["sub_question"]
        filters = {"article": payload["article_filter"]} if payload.get("article_filter") else None

        # the corpus is Polish — translate the retrieval query when needed
        if payload.get("query_language") == "en":
            retrieval_query = await services.translator.translate(sub_question, Direction.EN_TO_PL)
        else:
            retrieval_query = sub_question

        chunks = await services.search.retrieve(retrieval_query, top_k=_TOP_K, filters=filters)
        evidence = await compress_tool(services.llm, question=sub_question, chunks=chunks)

        thought = await think_tool(services.llm, question=sub_question, evidence=evidence)
        if not thought.sufficient and thought.refined_query:
            # one autonomous retry with the refined Polish query
            more_chunks = await services.search.retrieve(
                thought.refined_query, top_k=_TOP_K, filters=None
            )
            seen = {item.quote for item in evidence}
            extra = await compress_tool(services.llm, question=sub_question, chunks=more_chunks)
            evidence.extend(item for item in extra if item.quote not in seen)
            thought = await think_tool(services.llm, question=sub_question, evidence=evidence)

        logger.info(
            "sub_agent.done",
            sub_question=sub_question[:60],
            evidence=len(evidence),
            sufficient=thought.sufficient,
        )
        return {
            "sub_results": [
                SubAgentResult(
                    sub_question=sub_question,
                    evidence=evidence,
                    sufficient=thought.sufficient,
                )
            ]
        }

    return sub_agent
