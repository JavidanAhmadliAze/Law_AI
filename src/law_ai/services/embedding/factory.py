from law_ai.config import Settings
from law_ai.services.embedding.base import BaseEmbedder
from law_ai.services.embedding.client import APIEmbedder, LocalEmbedder


class EmbedderFactory:
    @staticmethod
    def create(settings: Settings) -> BaseEmbedder:
        if not settings.embedding.model:
            raise ValueError("EMBEDDING__MODEL is not set — configure it in .env")
        match settings.embedding.provider:
            case "local":
                return LocalEmbedder(settings.embedding)
            case "api":
                return APIEmbedder(settings.embedding)
            case other:
                raise ValueError(f"Unknown embedding provider: {other!r}")
