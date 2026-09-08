"""SSE streaming endpoint — verified with a fake astream graph (no LLM/infra).

Asserts token streaming, node filtering, the final/done control events, the
cache-hit JSON path, and persistence.
"""

import uuid

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from law_ai.dependencies import (
    get_agentic_rag,
    get_cache,
    get_conversation_repository,
    get_current_user,
)
from law_ai.exceptions import NotFoundError
from law_ai.main import create_app
from law_ai.schemas.chat import AskResponse

CHAT_ID = uuid.uuid4()
USER_ID = uuid.uuid4()


class FakeChat:
    id = CHAT_ID
    user_id = USER_ID
    title = "existing chat"


class FakeRepo:
    def __init__(self) -> None:
        self.added: list[tuple[str, str]] = []

    async def get_chat(self, chat_id, user_id):  # type: ignore[no-untyped-def]
        if chat_id != CHAT_ID or user_id != USER_ID:
            raise NotFoundError("Chat not found")
        return FakeChat()

    async def list_messages(self, chat_id):  # type: ignore[no-untyped-def]
        return []

    async def add_message(self, chat_id, role, content):  # type: ignore[no-untyped-def]
        self.added.append((role, content))

    async def update(self, chat_id, values):  # type: ignore[no-untyped-def]
        pass


ANSWER = "The dog keeper is liable (Art. 431)."


class StreamingGraph:
    """Mimics graph.astream(stream_mode=["messages"]) for the writer node."""

    def __init__(self) -> None:
        self.seen_inputs: list = []

    async def astream(self, inputs, config, stream_mode):  # type: ignore[no-untyped-def]
        self.seen_inputs.append(inputs)
        for tok in ["The ", "dog ", "keeper ", "is ", "liable (Art. 431)."]:
            yield "messages", (AIMessageChunk(content=tok), {"langgraph_node": "writer"})
        # a non-writer token that must be filtered out
        yield "messages", (AIMessageChunk(content="IGNORED"), {"langgraph_node": "guardian"})
        # LangGraph also emits the AIMessage the writer returns in `messages`
        # (on_chain_end). It is a full AIMessage, not a chunk, and must NOT be
        # appended again — otherwise the answer is duplicated.
        yield "messages", (AIMessage(content=ANSWER), {"langgraph_node": "writer"})


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


def _client(repo: FakeRepo, graph: StreamingGraph | None = None) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: type("U", (), {"id": USER_ID})()
    app.dependency_overrides[get_conversation_repository] = lambda: repo
    app.dependency_overrides[get_agentic_rag] = lambda: graph or StreamingGraph()
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
    assert kinds[-1] == "done"

    tokens = [d["text"] for e, d in events if e == "token"]
    assert "IGNORED" not in "".join(tokens)  # non-writer node filtered out
    # streamed as deltas, not one lump, and the node's own AIMessage is not
    # appended on top of them
    assert len(tokens) == 5
    assert "".join(tokens) == ANSWER

    # accumulated answer persisted to history
    assert ("assistant", ANSWER) in repo.added


def test_graph_receives_only_the_new_question() -> None:
    """Conversation memory is the checkpointer's job — the router must not
    replay the Postgres transcript, which would append id-less duplicates that
    `add_messages` cannot dedupe against the checkpoint."""
    repo = FakeRepo()
    graph = StreamingGraph()
    client = _client(repo, graph)
    with client.stream("POST", f"/chats/{CHAT_ID}/ask/stream", json={"question": "dog?"}) as resp:
        assert resp.status_code == 200
        "".join(resp.iter_text())

    assert len(graph.seen_inputs) == 1
    sent = graph.seen_inputs[0]["messages"]
    assert [m.content for m in sent] == ["dog?"]


class FakeCache:
    """Exact-match cache stub: records what was stored, serves what was seeded."""

    def __init__(self, seeded: AskResponse | None = None) -> None:
        self.seeded = seeded
        self.stored: list[AskResponse] = []

    async def find_cached_response(self, request):  # type: ignore[no-untyped-def]
        return self.seeded

    async def store_response(self, request, response):  # type: ignore[no-untyped-def]
        self.stored.append(response)


def _ask(client: TestClient) -> list[tuple[str, dict]]:
    with client.stream("POST", f"/chats/{CHAT_ID}/ask/stream", json={"question": "dog?"}) as resp:
        assert resp.status_code == 200
        return _parse_sse("".join(resp.iter_text()))


def test_cache_hit_returns_whole_answer_as_json_not_a_stream() -> None:
    """A cached answer already exists — there is nothing to stream, so it comes
    back as one JSON body and the graph is never touched."""
    repo = FakeRepo()
    graph = StreamingGraph()
    other_chat = uuid.uuid4()  # the chat that first cached this answer
    cache = FakeCache(seeded=AskResponse(answer=ANSWER, conversation_id=other_chat))
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: type("U", (), {"id": USER_ID})()
    app.dependency_overrides[get_conversation_repository] = lambda: repo
    app.dependency_overrides[get_agentic_rag] = lambda: graph
    app.dependency_overrides[get_cache] = lambda: cache

    resp = TestClient(app).post(f"/chats/{CHAT_ID}/ask/stream", json={"question": "dog?"})
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    body = resp.json()
    assert body["answer"] == ANSWER
    # rewritten to the caller's chat, not the one that first cached it
    assert body["conversation_id"] == str(CHAT_ID)
    assert graph.seen_inputs == []  # graph never invoked
    assert ("assistant", ANSWER) in repo.added


def test_cache_miss_streams_then_stores() -> None:
    repo = FakeRepo()
    graph = StreamingGraph()
    cache = FakeCache(seeded=None)
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: type("U", (), {"id": USER_ID})()
    app.dependency_overrides[get_conversation_repository] = lambda: repo
    app.dependency_overrides[get_agentic_rag] = lambda: graph
    app.dependency_overrides[get_cache] = lambda: cache

    with TestClient(app).stream(
        "POST", f"/chats/{CHAT_ID}/ask/stream", json={"question": "dog?"}
    ) as resp:
        assert "text/event-stream" in resp.headers["content-type"]
        events = _parse_sse("".join(resp.iter_text()))
    assert "".join(d["text"] for e, d in events if e == "token") == ANSWER
    assert [e for e, _ in events][-1] == "done"
    assert len(graph.seen_inputs) == 1
    # the answer is written back so the next identical question is a hit
    assert [r.answer for r in cache.stored] == [ANSWER]
