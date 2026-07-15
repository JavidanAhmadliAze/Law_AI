from law_ai.config import Settings
from law_ai.services.fetcher.base import BaseFetcher
from law_ai.services.fetcher.client import HttpFetcher


def create_fetcher(settings: Settings) -> BaseFetcher:  # noqa: ARG001 — uniform factory signature
    return HttpFetcher()
