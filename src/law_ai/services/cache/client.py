"""Redis-backed exact-match cache for RAG answers.

Cache errors are never fatal: a broken Redis degrades to cache misses and
the question flows through the normal pipeline.
"""

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis
from pydantic import ValidationError

from law_ai.config import RedisSettings
from law_ai.logging import get_logger
from law_ai.schemas.chat import AskRequest, AskResponse
from law_ai.services.cache.base import BaseCache

logger = get_logger(__name__)

_KEY_PREFIX = "exact_cache:"


class CacheClient(BaseCache):
    def __init__(
        self,
        redis_client: aioredis.Redis,
        settings: RedisSettings,
        *,
        pipeline_fingerprint: dict[str, Any],
    ) -> None:
        self._redis = redis_client
        self._settings = settings
        self._ttl_seconds = settings.ttl_hours * 3600
        # models/index that shape an answer — part of every key, so a model
        # swap or re-pointed index can never serve stale answers
        self._fingerprint = pipeline_fingerprint

    # ------------------------------------------------------------ lifecycle

    async def startup(self) -> None:
        await self._redis.ping()
        logger.info("cache.startup", ttl_hours=self._settings.ttl_hours)

    async def teardown(self) -> None:
        await self._redis.aclose()
        logger.info("cache.teardown")

    # ------------------------------------------------------------ cache ops

    def _generate_cache_key(self, request: AskRequest) -> str:
        payload = {
            "question": " ".join(request.question.lower().split()),  # normalize whitespace/case
            **self._fingerprint,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"{_KEY_PREFIX}{digest}"

    async def find_cached_response(self, request: AskRequest) -> AskResponse | None:
        key = self._generate_cache_key(request)
        try:
            raw = await self._redis.get(key)
        except aioredis.RedisError as exc:
            logger.warning("cache.get_failed", error=str(exc))
            return None
        if raw is None:
            return None
        try:
            response = AskResponse.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("cache.corrupted_entry", key=key, error=str(exc))
            return None
        logger.info("cache.hit", key=key)
        return response

    async def store_response(self, request: AskRequest, response: AskResponse) -> None:
        key = self._generate_cache_key(request)
        try:
            await self._redis.set(key, response.model_dump_json(), ex=self._ttl_seconds)
        except aioredis.RedisError as exc:
            logger.warning("cache.set_failed", error=str(exc))
            return
        logger.info("cache.stored", key=key, ttl_seconds=self._ttl_seconds)
