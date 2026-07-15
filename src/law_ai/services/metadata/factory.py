from law_ai.config import Settings
from law_ai.services.metadata.base import BaseMetadataBuilder
from law_ai.services.metadata.client import ActMetadataBuilder


def create_metadata_builder(settings: Settings) -> BaseMetadataBuilder:  # noqa: ARG001 — uniform factory signature
    return ActMetadataBuilder()
