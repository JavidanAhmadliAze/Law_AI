"""Supervisor — judges coverage with think-style reflection and either loops
(more sub-questions) or releases the answer to the writer."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from law_ai.logging import get_logger
from law_ai.services.agents import prompts
from law_ai.services.agents.context import services_from_config
from law_ai.services.agents.schemas import SupervisorDecision
from law_ai.services.agents.state import GraphState

logger = get_logger(__name__)

MAX_ITERATIONS = 2


async def supervisor(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    iterations = state.get("iterations", 0) + 1
    results = state.get("sub_results", [])

    if iterations >= MAX_ITERATIONS:
        # budget spent — write with what we have
        return {"iterations": iterations, "additional_questions": []}

    gathered = (
        "\n\n".join(
            f"Sub-question: {r.sub_question}\n"
            + "\n".join(f"- [{e.source_article}] {e.claim}" for e in r.evidence)
            + f"\n(sub-agent judged sufficient: {r.sufficient})"
            for r in results
        )
        or "(nothing gathered)"
    )

    decision = await services_from_config(config).llm.generate_structured(
        prompts.SUPERVISOR,
        f"User question: {state['question']}\n\nGathered evidence:\n{gathered}",
        SupervisorDecision,
    )
    additional = [] if decision.complete else decision.additional_questions[:2]
    logger.info(
        "supervisor.decision",
        complete=decision.complete,
        additional=len(additional),
        iteration=iterations,
    )
    return {"iterations": iterations, "additional_questions": additional}
