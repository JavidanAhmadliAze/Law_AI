"""Route-level tests for /ask — enabled by dependency injection.

The RAG stack (graph, cache, tracer) arrives through Depends(), so these
tests swap in fakes with `dependency_overrides` and exercise the endpoint's
branches without Postgres, Redis, OpenSearch or an LLM.

TestClient is used WITHOUT its context manager on purpose: that skips the
lifespan, so no real services boot and app.state stays empty (which is also
what lets the 503 case be exercised).
"""

import uuid
from typing import Any

from fastapi.testclient import TestClient

from law_ai.dependencies import (
    get_agentic_rag,
    get_cache,
    get_conversation_repository,
    get_current_user,
    get_langfuse,
)
from law_ai.main import create_app
from law_ai.schemas.chat import AskResponse, Citation

CHAT_ID = uuid.uuid4()
USER_ID = uuid.uuid4()


class FakeChat:
    id = CHAT_ID
    user_id = USER_ID
    title = "existing chat"


class FakeRepo:
    """Minimal ConversationRepository stand-in; records what was written."""

    def __init__(self, messages: list[Any] | None = None) -> None:
        self.messages = messages or []
        self.added: list[tuple[str, str]] = []

    async def get(self, chat_id: uuid.UUID) -> FakeChat | None:
        return FakeChat() if chat_id == CHAT_ID else None

    async def list_messages(self, chat_id: uuid.UUID) -> list[Any]:
        return self.messages

    async def add_message(self, chat_id: uuid.UUID, role: str, content: str) -> None:
        self.added.append((role, content))

    async def update(self, chat_id: uuid.UUID, values: dict) -> None:
        pass


class ExplodingGraph:
    """Fails the test if the pipeline runs when it should have been skipped."""

    async def ainvoke(self, *args: Any, **kwargs: Any) -> dict:
        raise AssertionError("graph must not run on a cache hit")


class HitCache:
    def __init__(self) -> None:
        self.stored: list[Any] = []

    async def find_cached_response(self, request: Any) -> AskResponse:
        return AskResponse(
            answer="cached answer",
            citations=[Citation(article="Art. 431", quote="Kto zwierzę chowa...")],
            conversation_id=CHAT_ID,
        )

    async def store_response(self, request: Any, response: Any) -> None:
        self.stored.append(response)


def _client(repo: FakeRepo, cache: Any, graph: Any) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: type("U", (), {"id": USER_ID})()
    app.dependency_overrides[get_conversation_repository] = lambda: repo
    app.dependency_overrides[get_agentic_rag] = lambda: graph
    app.dependency_overrides[get_cache] = lambda: cache
    app.dependency_overrides[get_langfuse] = lambda: None
    return TestClient(app)  # no `with`: lifespan never runs


def test_cache_hit_skips_the_graph() -> None:
    repo, cache = FakeRepo(), HitCache()
    client = _client(repo, cache, ExplodingGraph())
    response = client.post(f"/chats/{CHAT_ID}/ask", json={"question": "who is liable?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "cached answer"
    assert body["citations"][0]["article"] == "Art. 431"
    # the cached answer is still persisted to the conversation
    assert ("assistant", "cached answer") in repo.added


def test_missing_chat_is_404() -> None:
    client = _client(FakeRepo(), HitCache(), ExplodingGraph())
    response = client.post(f"/chats/{uuid.uuid4()}/ask", json={"question": "hi"})
    assert response.status_code == 404


def test_rag_unavailable_returns_503() -> None:
    """The graph dependency raises when the RAG stack failed to boot."""
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: type("U", (), {"id": USER_ID})()
    app.dependency_overrides[get_conversation_repository] = lambda: FakeRepo()
    app.dependency_overrides[get_cache] = lambda: None
    app.dependency_overrides[get_langfuse] = lambda: None
    # get_agentic_rag NOT overridden → reads app.state, which has no graph
    client = TestClient(app)
    response = client.post(f"/chats/{CHAT_ID}/ask", json={"question": "hi"})
    assert response.status_code == 503
