"""Provider-agnostic LLM client built on LangChain's init_chat_model.

`provider` + `model` come from env (LLM__PROVIDER / LLM__MODEL); any provider
LangChain supports (anthropic, bedrock, openai, ollama, ...) works unchanged.
"""

import asyncio
import json
from collections.abc import AsyncIterator
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

_STRUCTURED_ATTEMPTS = 3
_STRUCTURED_BACKOFF = 1.0  # seconds, multiplied by the attempt number


class LangChainLLM(BaseLLM):
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        kwargs: dict[str, Any] = {"temperature": settings.temperature}
        if settings.api_key:
            kwargs["api_key"] = settings.api_key
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        if settings.reasoning_effort:
            kwargs["reasoning_effort"] = settings.reasoning_effort
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

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        """Token deltas from `.astream` — this is what makes LangGraph's
        `stream_mode="messages"` fire `on_llm_new_token`, so the writer node's
        tokens reach the SSE router as they generate instead of in one lump at
        the end."""
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        try:
            async for chunk in self._chat.astream(messages):
                if text := chunk.text:
                    yield text
        except Exception as exc:  # provider errors normalized to domain error
            logger.error("llm.stream_failed", error=str(exc))
            raise GenerationError(f"LLM stream failed: {exc}") from exc

    async def generate_structured[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        method = self._settings.structured_method
        structured = self._chat.with_structured_output(schema, method=method)
        if method == "json_mode":
            system = (
                f"{system}\n\nRespond ONLY with a json object conforming to this "
                f"JSON Schema: {json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
            )

        last: Exception | None = None
        for attempt in range(1, _STRUCTURED_ATTEMPTS + 1):
            try:
                result = await structured.ainvoke(
                    [SystemMessage(content=system), HumanMessage(content=user)]
                )
            except Exception as exc:  # noqa: PERF203 — retry is the point
                last = exc
                logger.warning(
                    "llm.generate_structured_retry",
                    schema=schema.__name__,
                    attempt=attempt,
                    of=_STRUCTURED_ATTEMPTS,
                    error=str(exc)[:200],
                )
                if attempt < _STRUCTURED_ATTEMPTS:
                    await asyncio.sleep(_STRUCTURED_BACKOFF * attempt)
                continue
            return result  # type: ignore[return-value]

        logger.error("llm.generate_structured_failed", schema=schema.__name__, error=str(last))
        raise GenerationError(f"Structured LLM call failed after {_STRUCTURED_ATTEMPTS}: {last}")

    async def health_check(self) -> bool:
        return bool(self._settings.model)
