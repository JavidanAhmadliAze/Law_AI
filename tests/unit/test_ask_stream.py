"""SSE streaming endpoint — verified with a fake astream graph (no LLM/infra).

Citations are left for the agent to supply; this skeleton only asserts token
streaming, node filtering, the final/done control events, and persistence.
"""

import uuid

from fastapi.testclient import TestClient

from law_ai.dependencies import (
    get_agentic_rag,
    get_cache,
    get_conversation_repository,
    get_current_user,
)
from law_ai.main import create_app

CHAT_ID = uuid.uuid4()
USER_ID = uuid.uuid4()


class FakeChat:
    id = CHAT_ID
    user_id = USER_ID
    title = "existing chat"


class FakeRepo:
    def __init__(self) -> None:
        self.added: list[tuple[str, str]] = []

    async def get(self, chat_id):  # type: ignore[no-untyped-def]
        return FakeChat() if chat_id == CHAT_ID else None

    async def list_messages(self, chat_id):  # type: ignore[no-untyped-def]
        return []

    async def add_message(self, chat_id, role, content):  # type: ignore[no-untyped-def]
        self.added.append((role, content))

    async def update(self, chat_id, values):  # type: ignore[no-untyped-def]
        pass


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class StreamingGraph:
    """Mimics graph.astream(stream_mode=["messages"])."""

    async def astream(self, inputs, config, stream_mode):  # type: ignore[no-untyped-def]
        for tok in ["The ", "dog ", "keeper ", "is ", "liable (Art. 431)."]:
            yield "messages", (_Msg(tok), {"langgraph_node": "writer"})
        # a non-writer token that must be filtered out
        yield "messages", (_Msg("IGNORED"), {"langgraph_node": "guardian"})


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    import json

    out = []
    for block in text.strip().split("\n\n"):
        ev = da = None
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                da = json.loads(line[6:])
        if ev:
            out.append((ev, da))
    return out


def _client(repo: FakeRepo) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: type("U", (), {"id": USER_ID})()
    app.dependency_overrides[get_conversation_repository] = lambda: repo
    app.dependency_overrides[get_agentic_rag] = lambda: StreamingGraph()
    app.dependency_overrides[get_cache] = lambda: None
    return TestClient(app)


def test_stream_emits_writer_tokens_then_final() -> None:
    repo = FakeRepo()
    client = _client(repo)
    with client.stream("POST", f"/chats/{CHAT_ID}/ask/stream", json={"question": "dog?"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())

    events = _parse_sse(body)
    kinds = [e for e, _ in events]
    assert kinds[-2:] == ["final", "done"]

    tokens = [d["text"] for e, d in events if e == "token"]
    assert "IGNORED" not in "".join(tokens)  # non-writer node filtered out
    assert "".join(tokens) == "The dog keeper is liable (Art. 431)."

    # accumulated answer persisted to history
    assert ("assistant", "The dog keeper is liable (Art. 431).") in repo.added
