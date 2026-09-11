from law_ai.config import Settings
from law_ai.services.llm.base import BaseLLM
from law_ai.services.llm.client import LangChainLLM


def create_llm(settings: Settings) -> BaseLLM:
    """The generation model — used by the writer."""
    if not settings.llm.model:
        raise ValueError("LLM__MODEL is not set — configure it in .env")
    return LangChainLLM(settings.llm)


def create_fast_llm(settings: Settings) -> BaseLLM:
    """The cheap model for guardian / query rewriter / translator.

    Same provider and credentials, different model id. Falls back to the
    generation model when LLM__FAST_MODEL is unset, so the split is opt-in.
    """
    if not settings.llm.fast_model:
        return create_llm(settings)
    return LangChainLLM(settings.llm.model_copy(update={"model": settings.llm.fast_model}))
