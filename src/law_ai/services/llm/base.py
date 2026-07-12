"""LLM service contract.

Provider and model are opaque strings from Settings — nothing in the codebase
references a specific model. Structured output is first-class: agents rely on
typed results for deterministic graph routing.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, system: str, user: str) -> str:
        """Plain text completion."""

    @abstractmethod
    async def generate_structured[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        """Completion parsed/validated into the given pydantic schema."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Cheap availability probe (config sanity, not a paid call)."""
