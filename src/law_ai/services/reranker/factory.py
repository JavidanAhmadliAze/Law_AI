from typing import Any

from law_ai.config import Settings
from law_ai.services.reranker.base import BaseReranker
from law_ai.services.reranker.client import CrossEncoderReranker


def create_reranker(settings: Settings, tracer: Any = None) -> BaseReranker:
    return CrossEncoderReranker(settings.reranker, tracer=tracer)
