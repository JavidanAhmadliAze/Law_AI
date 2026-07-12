from law_ai.config import Settings
from law_ai.services.fetcher.base import BaseFetcher
from law_ai.services.fetcher.client import HttpFetcher


class FetcherFactory:
    @staticmethod
    def create(settings: Settings) -> BaseFetcher:  # noqa: ARG004 — uniform factory signature
        return HttpFetcher()
