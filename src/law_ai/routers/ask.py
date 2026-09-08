"""Ask endpoint — cached answers whole, fresh answers streamed.

POST /chats/{id}/ask/stream answers in one of two shapes, so clients must
branch on Content-Type:

- cache hit  → application/json, a complete AskResponse. There is nothing to
               stream: the answer already exists, so dribbling it out as fake
               tokens would only add latency.
- cache miss → text/event-stream: `token` events as the agent generates, then
               `done`. The accumulated answer is persisted and cached before
               `done` is sent, so a client that has seen it knows the turn is
               fully written.

Conversation memory is entirely the checkpointer's job: the graph is invoked
with thread_id=chat_id and only the new question, so this router carries no
notion of prior turns. The Postgres `messages` rows it writes are the UI's
transcript (GET /chats/{id}/messages), never agent context.

The agent is injected via `get_agentic_rag` (app.state.agentic_rag) — plug your
compiled graph in there. Until one is set, this endpoint returns 503. The cache
is a separate concern: it stores/serves AskResponse and has no dependency on the
agent internals.
"""

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage

from law_ai.dependencies import (
    AgentGraphDep,
    CacheDep,
    ConversationRepoDep,
    CurrentUserDep,
)
from law_ai.models.conversation import NEW_CHAT_TITLE
from law_ai.schemas.chat import AskRequest, AskResponse, ErrorResponse, TokenResponse

router = APIRouter(prefix="/chats", tags=["ask"])

_TITLE_LEN = 60


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/{chat_id}/ask/stream", response_model=None)
async def ask_stream(
    chat_id: uuid.UUID,
    payload: AskRequest,
    user: CurrentUserDep,
    chats: ConversationRepoDep,
    graph: AgentGraphDep,
    cache: CacheDep,
) -> AskResponse | StreamingResponse:
    """Cache hit → the whole AskResponse as JSON. Miss → an SSE token stream of
    `token` events, then `done`."""
    chat = await chats.get_chat(chat_id, user.id)  # 404s unless the caller owns it
    # Postgres holds the UI transcript only (GET /chats/{id}/messages); the
    # agent's conversation memory is the checkpointer, keyed on thread_id.
    await chats.add_message(chat_id, "user", payload.question)
    if chat.title == NEW_CHAT_TITLE:
        await chats.update(chat_id, {"title": payload.question[:_TITLE_LEN]})

    if cache is not None:
        cached = await cache.find_cached_response(payload)
        if cached is not None:
            await chats.add_message(chat_id, "assistant", cached.answer)
            # the stored conversation_id belongs to whichever chat first asked
            # this question — the caller wants their own
            return cached.model_copy(update={"conversation_id": chat_id})

    async def events() -> AsyncIterator[str]:
        # only the new question: the checkpointer supplies the rest of the thread
        config = {"configurable": {"thread_id": str(chat_id)}}
        answer_parts: list[str] = []
        try:
            async for mode, chunk in graph.astream(
                {"messages": [HumanMessage(content=payload.question)]},
                config=config,
                stream_mode=["messages"],
            ):
                if mode == "messages":
                    msg, meta = chunk
                    # Only the writer node, and only its token deltas. The
                    # AIMessageChunk check is load-bearing: LangGraph also emits
                    # the AIMessage the writer returns in `messages` (via
                    # on_chain_end), which would append the whole answer a
                    # second time after the stream.
                    if meta.get("langgraph_node") != "writer":
                        continue
                    if not isinstance(msg, AIMessageChunk):
                        continue
                    if text := msg.text:
                        answer_parts.append(text)
                        yield _sse("token", TokenResponse(text=text).model_dump())
        except Exception as exc:  # noqa: BLE001 — surface a clean stream error
            yield _sse("error", ErrorResponse(detail=str(exc)).model_dump())
            return

        answer = "".join(answer_parts)

        await chats.add_message(chat_id, "assistant", answer)
        # `answer` is empty when no writer tokens streamed (e.g. a guardian
        # refusal ends the graph early) — never cache that.
        if cache is not None and answer:
            await cache.store_response(
                payload,
                AskResponse(answer=answer, conversation_id=chat_id),
            )
        yield _sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")
