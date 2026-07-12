"""Translation service contract (EN↔PL).

Matters most for the sparse/BM25 leg of hybrid retrieval: an English token
never lexically matches a Polish article. The glossary guarantees canonical
Polish legal terminology; a model/LLM handles the surrounding sentence.
Quoted constitutional text used for citations is never translated.
"""

from abc import ABC, abstractmethod
from enum import StrEnum


class Direction(StrEnum):
    EN_TO_PL = "en->pl"
    PL_TO_EN = "pl->en"


class BaseTranslator(ABC):
    @abstractmethod
    async def detect_language(self, text: str) -> str:
        """Return ISO 639-1 code ('en', 'pl', ...)."""

    @abstractmethod
    async def translate(self, text: str, direction: Direction) -> str:
        """Translate text; glossary terms are applied deterministically first."""
