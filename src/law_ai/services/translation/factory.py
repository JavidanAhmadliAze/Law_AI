from law_ai.config import Settings
from law_ai.services.llm.base import BaseLLM
from law_ai.services.translation.base import BaseTranslator
from law_ai.services.translation.client import GlossaryOnlyTranslator, LLMTranslator


def create_translator(settings: Settings, llm: BaseLLM | None = None) -> BaseTranslator:
    match settings.translation.provider:
        case "glossary-only":
            return GlossaryOnlyTranslator()
        case "llm":
            if llm is None:
                raise ValueError("create_translator needs the llm service for provider='llm'")
            return LLMTranslator(llm)
        case other:
            raise ValueError(f"Unknown translation provider: {other!r}")
