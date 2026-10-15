"""Structured-output retry: a formatting slip must not lose the caller's work."""

from typing import Any

import pytest
from pydantic import BaseModel

from law_ai.config import LLMSettings
from law_ai.exceptions import GenerationError
from law_ai.services.llm.client import LangChainLLM


class _Shape(BaseModel):
    value: str


class _FlakyStructured:
    """Fails `failures` times, then returns a valid object."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def ainvoke(self, _messages: Any) -> _Shape:
        self.calls += 1
        if self.calls <= self.failures:
            raise ValueError("Invalid json output: <stray token>")
        return _Shape(value="ok")


def _llm(stub: _FlakyStructured) -> LangChainLLM:
    llm = LangChainLLM.__new__(LangChainLLM)
    llm._settings = LLMSettings(model="test", structured_method="json_mode")  # type: ignore[attr-defined]
    llm._chat = type("_Chat", (), {"with_structured_output": lambda *a, **k: stub})()  # type: ignore[attr-defined]
    return llm


@pytest.mark.asyncio
async def test_retries_past_a_transient_parse_failure() -> None:
    stub = _FlakyStructured(failures=1)
    result = await _llm(stub).generate_structured("sys", "user", _Shape)
    assert result.value == "ok"
    assert stub.calls == 2  # failed once, succeeded on the retry


@pytest.mark.asyncio
async def test_gives_up_after_the_attempt_budget() -> None:
    stub = _FlakyStructured(failures=99)
    with pytest.raises(GenerationError):
        await _llm(stub).generate_structured("sys", "user", _Shape)
    assert stub.calls == 3
