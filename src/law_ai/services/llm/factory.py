from law_ai.config import Settings
from law_ai.services.llm.base import BaseLLM
from law_ai.services.llm.client import LangChainLLM


class LLMFactory:
    @staticmethod
    def create(settings: Settings) -> BaseLLM:
        if not settings.llm.model:
            raise ValueError("LLM__MODEL is not set — configure it in .env")
        return LangChainLLM(settings.llm)
