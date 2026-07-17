import redis.asyncio as aioredis

from law_ai.config import Settings
from law_ai.logging import get_logger
from law_ai.services.cache.base import BaseCache
from law_ai.services.cache.client import CacheClient

logger = get_logger(__name__)


def create_redis_client(settings: Settings) -> aioredis.Redis:
    """Configured connection only — liveness is verified by CacheClient.startup()."""
    r = settings.redis
    return aioredis.Redis(
        host=r.host,
        port=r.port,
        password=r.password or None,
        db=r.db,
        decode_responses=True,
        socket_timeout=r.socket_timeout_seconds,
        socket_connect_timeout=r.socket_connect_timeout_seconds,
        retry_on_timeout=True,
        retry_on_error=[aioredis.ConnectionError, aioredis.TimeoutError],
    )


def create_cache_client(settings: Settings) -> BaseCache:
    if not settings.redis.enabled:
        raise ValueError("REDIS__ENABLED=false — cache disabled by config")
    client = CacheClient(
        create_redis_client(settings),
        settings.redis,
        # everything that shapes an answer belongs in the key
        pipeline_fingerprint={
            "llm": f"{settings.llm.provider}:{settings.llm.model}",
            "embedding": settings.embedding.model,
            "reranker": settings.reranker.model,
            "index": settings.opensearch.index,
        },
    )
    logger.info("cache.client_created", host=settings.redis.host, port=settings.redis.port)
    return client
