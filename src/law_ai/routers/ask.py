"""Ask endpoint — streaming (SSE) over the agent graph.

POST /chats/{id}/ask/stream streams the agent's tokens as they generate
(text/event-stream), then a `final` event (citations) and `done`. The
accumulated answer is persisted to history and cached once the stream completes.

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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from law_ai.dependencies import (
    AgentGraphDep,
    CacheDep,
    ConversationRepoDep,
    CurrentUserDep,
)
from law_ai.exceptions import NotFoundError
from law_ai.schemas.chat import AskRequest, AskResponse, Citation, ErrorResponse, TokenResponse

router = APIRouter(prefix="/chats", tags=["ask"])

_HISTORY_TURNS = 6
_TITLE_LEN = 60


async def _prepare(chats, user, chat_id, payload):  # type: ignore[no-untyped-def]
    """Shared preamble: ownership check, history, persist user message, title."""
    chat = await chats.get(chat_id)
    if chat is None or chat.user_id != user.id:
        raise NotFoundError("Chat not found")
    history = [
        {"role": m.role, "content": m.content}
        for m in (await chats.list_messages(chat_id))[-_HISTORY_TURNS:]
    ]
    await chats.add_message(chat_id, "user", payload.question)
    if chat.title == "New chat":
        await chats.update(chat_id, {"title": payload.question[:_TITLE_LEN]})
    return history


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/{chat_id}/ask/stream")
async def ask_stream(
    chat_id: uuid.UUID,
    payload: AskRequest,
    user: CurrentUserDep,
    chats: ConversationRepoDep,
    graph: AgentGraphDep,
    cache: CacheDep,
) -> StreamingResponse:
    """SSE: `token` events as the agent generates, then `final` (citations) + `done`."""
    history = await _prepare(chats, user, chat_id, payload)

    async def events() -> AsyncIterator[str]:
        # cache hit → replay the stored answer as one token, then final
        if cache is not None and not history:
            cached = await cache.find_cached_response(payload)
            if cached is not None:
                yield _sse("token", TokenResponse(text=cached.answer).model_dump())
                yield _sse(
                    "final",
                    {"citations": [c.model_dump() for c in cached.citations], "cached": True},
                )
                await chats.add_message(chat_id, "assistant", cached.answer)
                yield _sse("done", {})
                return

        config = {"configurable": {"thread_id": str(chat_id)}}
        messages: list[BaseMessage] = [
            HumanMessage(content=m["content"])
            if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in history
        ]
        messages.append(HumanMessage(content=payload.question))
        answer_parts: list[str] = []
        citations: list[Citation] = []  # TODO: populate from your agent's output
        try:
            async for mode, chunk in graph.astream(
                {"messages": messages},
                config=config,
                stream_mode=["messages"],
            ):
                if mode == "messages":
                    msg, meta = chunk
                    # stream only the final writer node's tokens (rename to your node)
                    if meta.get("langgraph_node") == "writer" and getattr(msg, "content", ""):
                        answer_parts.append(msg.content)
                        yield _sse("token", TokenResponse(text=msg.content).model_dump())
        except Exception as exc:  # noqa: BLE001 — surface a clean stream error
            yield _sse("error", ErrorResponse(detail=str(exc)).model_dump())
            return

        answer = "".join(answer_parts)
        yield _sse("final", {"citations": [c.model_dump() for c in citations], "cached": False})

        await chats.add_message(chat_id, "assistant", answer)
        if cache is not None and not history and citations:
            await cache.store_response(
                payload,
                AskResponse(answer=answer, citations=citations, conversation_id=chat_id),
            )
        yield _sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")
