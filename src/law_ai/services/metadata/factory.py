from law_ai.config import Settings
from law_ai.services.metadata.base import BaseMetadataBuilder
from law_ai.services.metadata.client import ActMetadataBuilder


class MetadataBuilderFactory:
    @staticmethod
    def create(settings: Settings) -> BaseMetadataBuilder:  # noqa: ARG004 — uniform factory signature
        return ActMetadataBuilder()
