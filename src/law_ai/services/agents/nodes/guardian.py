"""Guardian — entry gate: security first, then relevance.

Emits a typed GuardianVerdict; when blocked it also writes the final_answer
so the graph can end immediately with a polite refusal.
"""

from collections.abc import Awaitable, Callable

from law_ai.logging import get_logger
from law_ai.services.agents import prompts
from law_ai.services.agents.context import AgentServices
from law_ai.services.agents.schemas import FinalAnswer, GuardianVerdict
from law_ai.services.agents.state import GraphState

logger = get_logger(__name__)

# fast deterministic pre-filter for blatant injection attempts
_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all instructions",
    "system prompt",
    "you are now",
    "developer mode",
)


def build_guardian(services: AgentServices) -> Callable[[GraphState], Awaitable[dict]]:
    async def guardian(state: GraphState) -> dict:
        question = state["question"]

        lowered = question.lower()
        if any(marker in lowered for marker in _INJECTION_MARKERS):
            verdict = GuardianVerdict(
                allowed=False,
                reason="injection",
                message="I can only answer questions about the Polish Constitution.",
            )
        else:
            history = _render_history(state.get("history", []))
            verdict = await services.llm.generate_structured(
                prompts.GUARDIAN,
                f"Conversation so far:\n{history}\n\nUser message:\n{question}",
                GuardianVerdict,
            )

        update: dict = {"guardian_verdict": verdict}
        if not verdict.allowed:
            logger.warning("guardian.blocked", reason=verdict.reason)
            update["final_answer"] = FinalAnswer(answer=verdict.message, citations=[])
        return update

    return guardian


def _render_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(new conversation)"
    return "\n".join(f"{turn['role']}: {turn['content'][:300]}" for turn in history[-6:])
