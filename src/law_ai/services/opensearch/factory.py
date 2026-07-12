from typing import Any

from law_ai.config import Settings
from law_ai.services.embedding.base import BaseEmbedder
from law_ai.services.opensearch.base import BaseSearchService
from law_ai.services.opensearch.client import OpenSearchService


class SearchServiceFactory:
    @staticmethod
    def create(settings: Settings, embedder: BaseEmbedder, tracer: Any = None) -> BaseSearchService:
        return OpenSearchService(settings.opensearch, settings.reranker, embedder, tracer=tracer)
