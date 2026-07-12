"""Translator implementations.

LLMTranslator: glossary first (deterministic legal terms), then the LLM
service translates the full sentence, instructed to keep the already-inserted
Polish terms untouched. Language detection is a cheap heuristic (Polish
diacritics + stopwords) — swap for a model if it ever misfires.
"""

import re

from law_ai.exceptions import TranslationError
from law_ai.logging import get_logger
from law_ai.services.llm.base import BaseLLM
from law_ai.services.translation.base import BaseTranslator, Direction
from law_ai.services.translation.glossary import apply_glossary

logger = get_logger(__name__)

_POLISH_CHARS = set("ąćęłńóśźż")
_POLISH_STOPWORDS = {"jest", "nie", "czy", "jak", "się", "oraz", "który", "która", "prawo"}

_SYSTEM_PROMPT = {
    Direction.EN_TO_PL: (
        "You are a legal translator. Translate the user's text from English to Polish. "
        "The text may already contain Polish legal terms — keep them EXACTLY as they are. "
        "Use precise Polish legal terminology. Return ONLY the translation."
    ),
    Direction.PL_TO_EN: (
        "You are a legal translator. Translate the user's text from Polish to English. "
        "Keep citations to articles (e.g. 'Art. 54') unchanged. "
        "Use precise legal terminology. Return ONLY the translation."
    ),
}


class LLMTranslator(BaseTranslator):
    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def detect_language(self, text: str) -> str:
        lowered = text.lower()
        if _POLISH_CHARS & set(lowered):
            return "pl"
        words = set(re.findall(r"\w+", lowered))
        if words & _POLISH_STOPWORDS:
            return "pl"
        return "en"

    async def translate(self, text: str, direction: Direction) -> str:
        if direction is Direction.EN_TO_PL:
            # deterministic legal terms first — they feed the BM25 leg verbatim
            text = apply_glossary(text)
        try:
            return await self._llm.generate(_SYSTEM_PROMPT[direction], text)
        except Exception as exc:
            logger.error("translation.failed", direction=str(direction), error=str(exc))
            raise TranslationError(f"Translation failed: {exc}") from exc


class GlossaryOnlyTranslator(BaseTranslator):
    """No-LLM fallback: glossary substitution only. Useful for tests/offline."""

    async def detect_language(self, text: str) -> str:
        return "pl" if _POLISH_CHARS & set(text.lower()) else "en"

    async def translate(self, text: str, direction: Direction) -> str:
        return apply_glossary(text) if direction is Direction.EN_TO_PL else text
