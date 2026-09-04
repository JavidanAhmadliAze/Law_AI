"""LLM service contract.

Provider and model are opaque strings from Settings — nothing in the codebase
references a specific model. Structured output is first-class: agents rely on
typed results for deterministic graph routing.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseLLM(ABC):
    @property
    @abstractmethod
    def chat_model(self) -> Any:
        """Underlying chat model — for tool-bound / streaming agent calls
        (e.g. `.bind_tools(...)`). Typed as Any to keep this contract free of a
        specific SDK; concrete clients return their provider's chat model."""

    @abstractmethod
    async def generate(self, system: str, user: str) -> str:
        """Plain text completion."""

    @abstractmethod
    async def generate_structured[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        """Completion parsed/validated into the given pydantic schema."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Cheap availability probe (config sanity, not a paid call)."""
