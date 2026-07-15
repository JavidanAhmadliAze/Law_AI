from law_ai.config import Settings
from law_ai.services.chunking.base import BaseChunker
from law_ai.services.chunking.client import ArticleChunker


def create_chunker(settings: Settings) -> BaseChunker:  # noqa: ARG001 — uniform factory signature
    return ArticleChunker()
