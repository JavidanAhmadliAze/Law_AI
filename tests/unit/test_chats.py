"""Chats router — ownership is the thing worth testing.

Conversation id == LangGraph thread id, so a leaky ownership check would let an
authenticated user read (or stream into) another user's thread. Every per-chat
route goes through `ConversationRepository.get_chat`, which scopes the query by
owner and answers 404 rather than 403 so foreign chat ids stay unprobeable.
"""

import uuid

from fastapi.testclient import TestClient

from law_ai.dependencies import get_conversation_repository, get_current_user
from law_ai.exceptions import NotFoundError
from law_ai.main import create_app

OWNER_ID = uuid.uuid4()
OTHER_ID = uuid.uuid4()
MY_CHAT = uuid.uuid4()
THEIR_CHAT = uuid.uuid4()


class FakeChat:
    def __init__(self, chat_id: uuid.UUID, user_id: uuid.UUID) -> None:
        self.id = chat_id
        self.user_id = user_id
        self.title = "Dog liability"
        self.created_at = __import__("datetime").datetime(2026, 1, 1)


class FakeRepo:
    def __init__(self) -> None:
        self.chats = {
            MY_CHAT: FakeChat(MY_CHAT, OWNER_ID),
            THEIR_CHAT: FakeChat(THEIR_CHAT, OTHER_ID),
        }
        self.messages: dict[uuid.UUID, list[dict]] = {}
        self.deleted: list[uuid.UUID] = []
        self.swept: list[uuid.UUID] = []
        self.created: list[FakeChat] = []

    async def get_chat(self, chat_id, user_id):  # type: ignore[no-untyped-def]
        """Mirrors the repository's owner-scoped query: a row that is not the
        caller's is simply not found."""
        chat = self.chats.get(chat_id)
        if chat is None or chat.user_id != user_id:
            raise NotFoundError("Chat not found")
        return chat

    async def list_messages(self, chat_id):  # type: ignore[no-untyped-def]
        return self.messages.get(chat_id, [])

    async def delete(self, chat_id):  # type: ignore[no-untyped-def]
        self.deleted.append(chat_id)

    async def delete_empty(self, user_id):  # type: ignore[no-untyped-def]
        empty = [
            c
            for c, chat in self.chats.items()
            if chat.user_id == user_id and not self.messages.get(c)
        ]
        self.swept += empty
        return len(empty)

    async def create(self, data):  # type: ignore[no-untyped-def]
        chat = FakeChat(uuid.uuid4(), data["user_id"])
        chat.title = data["title"]
        self.created.append(chat)
        return chat


def _client(repo: FakeRepo) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: type("U", (), {"id": OWNER_ID})()
    app.dependency_overrides[get_conversation_repository] = lambda: repo
    return TestClient(app)


def test_get_chat_returns_own_conversation() -> None:
    resp = _client(FakeRepo()).get(f"/chats/{MY_CHAT}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(MY_CHAT)
    assert body["title"] == "Dog liability"


def test_get_chat_hides_another_users_conversation_as_404() -> None:
    # 404, never 403 — a 403 would confirm the id exists
    resp = _client(FakeRepo()).get(f"/chats/{THEIR_CHAT}")
    assert resp.status_code == 404


def test_get_chat_missing_is_404() -> None:
    resp = _client(FakeRepo()).get(f"/chats/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_delete_refuses_another_users_conversation() -> None:
    repo = FakeRepo()
    assert _client(repo).delete(f"/chats/{THEIR_CHAT}").status_code == 404
    assert repo.deleted == []  # nothing removed


def test_messages_refuses_another_users_conversation() -> None:
    assert _client(FakeRepo()).get(f"/chats/{THEIR_CHAT}/messages").status_code == 404


def test_create_chat_sweeps_the_users_abandoned_empty_chats() -> None:
    """The UI creates a row on every "New chat" click, so a user who never asks
    anything would otherwise accumulate dead, untitled conversations."""
    repo = FakeRepo()
    resp = _client(repo).post("/chats", json={})
    assert resp.status_code == 201
    # swept before the new row is created, or we would delete the one just made
    assert repo.swept == [MY_CHAT]
    assert len(repo.created) == 1
    assert repo.created[0].id not in repo.swept
    assert resp.json()["title"] == "New chat"


def test_list_messages_returns_the_inline_transcript() -> None:
    """Messages live in the conversation row now; the endpoint shape is unchanged."""
    repo = FakeRepo()
    repo.messages[MY_CHAT] = [
        {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": "Kto odpowiada za psa?",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": "Art. 431 KC.",
            "created_at": "2026-01-01T00:00:05+00:00",
        },
    ]
    resp = _client(repo).get(f"/chats/{MY_CHAT}/messages")
    assert resp.status_code == 200
    assert [(m["role"], m["content"]) for m in resp.json()] == [
        ("user", "Kto odpowiada za psa?"),
        ("assistant", "Art. 431 KC."),
    ]


def test_create_chat_keeps_a_conversation_that_has_messages() -> None:
    repo = FakeRepo()
    repo.messages[MY_CHAT] = [
        {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": "q",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    assert _client(repo).post("/chats", json={}).status_code == 201
    assert repo.swept == []  # a chat with a transcript is never swept
