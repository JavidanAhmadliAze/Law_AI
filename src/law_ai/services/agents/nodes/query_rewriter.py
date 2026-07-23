"""Query rewriter — resolves references, decomposes, routes simple|complex."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from law_ai.logging import get_logger
from law_ai.services.agents import prompts
from law_ai.services.agents.context import services_from_config
from law_ai.services.agents.nodes.guardian import _render_history
from law_ai.services.agents.schemas import RewrittenQuery
from law_ai.services.agents.state import GraphState

logger = get_logger(__name__)


async def query_rewriter(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    question = state["question"]
    language = await services.translator.detect_language(question)

    rewritten = await services.llm.generate_structured(
        prompts.QUERY_REWRITER,
        f"Conversation so far:\n{_render_history(state.get('history', []))}\n\n"
        f"Question:\n{question}",
        RewrittenQuery,
    )
    sub_questions = rewritten.sub_questions or [question]
    logger.info(
        "query_rewriter.done",
        route=rewritten.route,
        sub_questions=len(sub_questions),
        language=language,
    )
    return {
        "sub_questions": sub_questions,
        "article_filter": rewritten.article_filter,
        "query_language": language,
        "iterations": 0,
    }
