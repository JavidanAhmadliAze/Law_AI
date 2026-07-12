"""Provider-agnostic LLM client built on LangChain's init_chat_model.

`provider` + `model` come from env (LLM__PROVIDER / LLM__MODEL); any provider
LangChain supports (anthropic, bedrock, openai, ollama, ...) works unchanged.
"""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from law_ai.config import LLMSettings
from law_ai.exceptions import GenerationError
from law_ai.logging import get_logger
from law_ai.services.llm.base import BaseLLM

logger = get_logger(__name__)


class LangChainLLM(BaseLLM):
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        kwargs: dict[str, Any] = {"temperature": settings.temperature}
        if settings.api_key:
            kwargs["api_key"] = settings.api_key
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._chat: BaseChatModel = init_chat_model(
            settings.model, model_provider=settings.provider or None, **kwargs
        )

    @property
    def chat_model(self) -> BaseChatModel:
        """Raw LangChain model — used by the agent graph for bound calls."""
        return self._chat

    async def generate(self, system: str, user: str) -> str:
        try:
            result = await self._chat.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        except Exception as exc:  # provider errors normalized to domain error
            logger.error("llm.generate_failed", error=str(exc))
            raise GenerationError(f"LLM call failed: {exc}") from exc
        return str(result.content)

    async def generate_structured[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        structured = self._chat.with_structured_output(schema)
        try:
            result = await structured.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        except Exception as exc:
            logger.error("llm.generate_structured_failed", schema=schema.__name__, error=str(exc))
            raise GenerationError(f"Structured LLM call failed: {exc}") from exc
        return result  # type: ignore[return-value]

    async def health_check(self) -> bool:
        return bool(self._settings.model)
