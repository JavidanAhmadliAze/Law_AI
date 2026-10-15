from law_ai.config import Settings
from law_ai.services.llm.base import BaseLLM
from law_ai.services.llm.client import LangChainLLM


def create_llm(settings: Settings) -> BaseLLM:
    """The generation model — used by the writer."""
    if not settings.llm.model:
        raise ValueError("LLM__MODEL is not set — configure it in .env")
    return LangChainLLM(settings.llm.model_copy(update={"reasoning_effort": ""}))


def create_fast_llm(settings: Settings) -> BaseLLM:
    """The cheap model for the guardian.

    Same provider and credentials, different model id. Falls back to the
    generation model when LLM__FAST_MODEL is unset, so the split is opt-in.
    """
    if not settings.llm.fast_model:
        return create_llm(settings)
    return LangChainLLM(
        settings.llm.model_copy(update={"model": settings.llm.fast_model, "reasoning_effort": ""})
    )


def create_reasoning_llm(settings: Settings) -> BaseLLM:
    """The reasoning model — used by the query rewriter.

    Decomposing a question into its legal functions is the one step where a
    wrong call is unrecoverable: a search aimed at the wrong institution cannot
    be rescued by retrieval or reranking downstream. Falls back to the fast
    model when LLM__REASONING_MODEL is unset, so the split stays opt-in.
    """
    if not settings.llm.reasoning_model:
        return create_fast_llm(settings)
    return LangChainLLM(settings.llm.model_copy(update={"model": settings.llm.reasoning_model}))
