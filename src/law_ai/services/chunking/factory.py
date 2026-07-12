from law_ai.config import Settings
from law_ai.services.chunking.base import BaseChunker
from law_ai.services.chunking.client import ArticleChunker


class ChunkerFactory:
    @staticmethod
    def create(settings: Settings) -> BaseChunker:  # noqa: ARG004 — uniform factory signature
        return ArticleChunker()
