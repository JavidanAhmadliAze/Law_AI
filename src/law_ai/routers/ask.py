"""POST /chats/{chat_id}/ask — run one question through the agentic RAG graph."""

import uuid
from contextlib import nullcontext

from fastapi import APIRouter

from law_ai.dependencies import (
    AgentGraphDep,
    CacheDep,
    ConversationRepoDep,
    CurrentUserDep,
    LangfuseDep,
)
from law_ai.exceptions import NotFoundError
from law_ai.schemas.chat import AskRequest, AskResponse, Citation
from law_ai.services.agents.schemas import FinalAnswer

router = APIRouter(prefix="/chats", tags=["ask"])

_HISTORY_TURNS = 6
_TITLE_LEN = 60


@router.post("/{chat_id}/ask", response_model=AskResponse)
async def ask(
    chat_id: uuid.UUID,
    payload: AskRequest,
    user: CurrentUserDep,
    chats: ConversationRepoDep,
    graph: AgentGraphDep,
    cache: CacheDep,
    langfuse: LangfuseDep,
) -> AskResponse:
    chat = await chats.get(chat_id)
    if chat is None or chat.user_id != user.id:
        raise NotFoundError("Chat not found")

    history = [
        {"role": m.role, "content": m.content}
        for m in (await chats.list_messages(chat_id))[-_HISTORY_TURNS:]
    ]
    await chats.add_message(chat_id, "user", payload.question)
    if chat.title == "New chat":  # first question names the chat
        await chats.update(chat_id, {"title": payload.question[:_TITLE_LEN]})

    # exact-match answer cache — only for history-free questions: a follow-up
    # with identical text could legitimately deserve a context-aware answer
    if cache is not None and not history:
        cached = await cache.find_cached_response(payload)
        if cached is not None:
            await chats.add_message(chat_id, "assistant", cached.answer)
            return AskResponse(
                answer=cached.answer, citations=cached.citations, conversation_id=chat_id
            )

    handler = langfuse.callback_handler() if langfuse else None
    config: dict = {"configurable": {"thread_id": str(chat_id)}}
    if handler is not None:
        config["callbacks"] = [handler]

    # one root span so the LangGraph callback spans AND the manual retrieval
    # spans nest into a single trace instead of splitting into two
    span_cm = (
        langfuse.span("ask", input={"question": payload.question, "chat_id": str(chat_id)})
        if langfuse
        else nullcontext()
    )
    with span_cm as root_span:
        result = await graph.ainvoke(
            {"question": payload.question, "history": history}, config=config
        )
        final: FinalAnswer = result.get("final_answer") or FinalAnswer(
            answer="I could not produce an answer — please try rephrasing.", citations=[]
        )
        if root_span is not None:
            root_span.update(output={"answer": final.answer, "citations": len(final.citations)})

    await chats.add_message(chat_id, "assistant", final.answer)
    response = AskResponse(
        answer=final.answer,
        citations=[Citation(article=c.article, quote=c.quote) for c in final.citations],
        conversation_id=chat_id,
    )
    if cache is not None and not history and final.citations:
        # don't memoize refusals/empty answers — only grounded results
        await cache.store_response(payload, response)
    return response
