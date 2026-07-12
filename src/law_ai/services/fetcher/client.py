import httpx

from law_ai.exceptions import FetchError
from law_ai.logging import get_logger
from law_ai.services.fetcher.base import BaseFetcher

logger = get_logger(__name__)


class HttpFetcher(BaseFetcher):
    def __init__(self, timeout: float = 60.0) -> None:
        self._timeout = timeout

    async def fetch(self, url: str) -> bytes:
        # sejm.gov.pl fronts requests with a bot check that rejects bare clients
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            )
        }
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True, headers=headers
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            # ISAP serves an HTML bot-challenge page under HTTP 200 when it
            # rate-limits — parsing it would silently index garbage
            raise FetchError(
                f"Expected a PDF from {url}, got "
                f"{response.headers.get('content-type', 'unknown')} "
                f"({len(response.content)} bytes) — likely a bot challenge; retry later"
            )
        logger.info("fetcher.downloaded", url=url, bytes=len(response.content))
        return response.content
