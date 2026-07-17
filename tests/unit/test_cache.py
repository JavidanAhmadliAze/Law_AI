import uuid

import redis.asyncio as aioredis

from law_ai.config import RedisSettings
from law_ai.schemas.chat import AskRequest, AskResponse, Citation
from law_ai.services.cache.client import CacheClient

_FINGERPRINT = {"llm": "deepseek:deepseek-chat", "embedding": "BAAI/bge-m3"}


class FakeRedis:
    """In-memory stand-in exposing the redis.asyncio surface CacheClient uses."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False

    async def get(self, key: str) -> str | None:
        if self.fail:
            raise aioredis.ConnectionError("down")
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self.fail:
            raise aioredis.ConnectionError("down")
        self.store[key] = value
        self.ttls[key] = ex or 0

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


def _client(fake: FakeRedis | None = None) -> tuple[CacheClient, FakeRedis]:
    fake = fake or FakeRedis()
    client = CacheClient(fake, RedisSettings(ttl_hours=2), pipeline_fingerprint=_FINGERPRINT)  # type: ignore[arg-type]
    return client, fake


def _response(answer: str = "Art. 659 KC says...") -> AskResponse:
    return AskResponse(
        answer=answer,
        citations=[Citation(article="Art. 659", quote="Przez umowę najmu...")],
        conversation_id=uuid.uuid4(),
    )


def test_key_is_deterministic_and_normalized() -> None:
    client, _ = _client()
    a = client._generate_cache_key(AskRequest(question="What is a lease?"))
    b = client._generate_cache_key(AskRequest(question="  what   is a LEASE?  "))
    assert a == b  # case + whitespace normalized
    assert a.startswith("exact_cache:")


def test_different_question_different_key() -> None:
    client, _ = _client()
    a = client._generate_cache_key(AskRequest(question="What is a lease?"))
    b = client._generate_cache_key(AskRequest(question="What is a mortgage?"))
    assert a != b


def test_fingerprint_changes_key() -> None:
    fake = FakeRedis()
    c1 = CacheClient(fake, RedisSettings(), pipeline_fingerprint={"llm": "a"})  # type: ignore[arg-type]
    c2 = CacheClient(fake, RedisSettings(), pipeline_fingerprint={"llm": "b"})  # type: ignore[arg-type]
    q = AskRequest(question="same question")
    assert c1._generate_cache_key(q) != c2._generate_cache_key(q)  # model swap → new keyspace


async def test_store_then_find_roundtrip() -> None:
    client, fake = _client()
    request = AskRequest(question="Who inherits first?")
    await client.store_response(request, _response())
    assert fake.ttls[client._generate_cache_key(request)] == 2 * 3600  # ttl_hours applied

    found = await client.find_cached_response(request)
    assert found is not None
    assert found.answer.startswith("Art. 659")
    assert found.citations[0].article == "Art. 659"


async def test_miss_returns_none() -> None:
    client, _ = _client()
    assert await client.find_cached_response(AskRequest(question="never asked")) is None


async def test_corrupted_entry_returns_none() -> None:
    client, fake = _client()
    request = AskRequest(question="broken")
    fake.store[client._generate_cache_key(request)] = "{not valid json"
    assert await client.find_cached_response(request) is None


async def test_redis_errors_degrade_to_miss() -> None:
    client, fake = _client()
    fake.fail = True
    request = AskRequest(question="redis is down")
    assert await client.find_cached_response(request) is None  # no exception escapes
    await client.store_response(request, _response())  # store swallows the error too
