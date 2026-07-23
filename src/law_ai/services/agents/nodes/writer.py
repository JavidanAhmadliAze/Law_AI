"""Writer — composes the grounded, cited final answer via the llm service."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from law_ai.logging import get_logger
from law_ai.services.agents import prompts
from law_ai.services.agents.context import services_from_config
from law_ai.services.agents.schemas import FinalAnswer
from law_ai.services.agents.state import GraphState

logger = get_logger(__name__)


async def writer(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    evidence_brief = (
        "\n\n".join(
            f"Sub-question: {r.sub_question}\n"
            + "\n".join(
                f"- claim: {e.claim}\n  source: {e.source_article}\n  quote: „{e.quote}”"
                for e in r.evidence
            )
            for r in state.get("sub_results", [])
        )
        or "(no evidence was retrieved)"
    )

    answer = await services_from_config(config).llm.generate_structured(
        prompts.WRITER,
        f"User question ({state.get('query_language', 'en')}): {state['question']}\n\n"
        f"Evidence:\n{evidence_brief}",
        FinalAnswer,
    )
    logger.info("writer.done", citations=len(answer.citations))
    return {"final_answer": answer}
