"""POST /chats/{chat_id}/ask — run one question through the agentic RAG graph."""

import uuid
from contextlib import nullcontext

from fastapi import APIRouter, Request

from law_ai.dependencies import ConversationRepoDep, CurrentUserDep
from law_ai.exceptions import LawAIError, NotFoundError
from law_ai.schemas.chat import AskRequest, AskResponse, Citation
from law_ai.services.agents.schemas import FinalAnswer

router = APIRouter(prefix="/chats", tags=["ask"])

_HISTORY_TURNS = 6
_TITLE_LEN = 60


class RAGUnavailableError(LawAIError):
    status_code = 503
    detail = "RAG pipeline is not configured (set LLM__MODEL and EMBEDDING__MODEL)"


@router.post("/{chat_id}/ask", response_model=AskResponse)
async def ask(
    chat_id: uuid.UUID,
    payload: AskRequest,
    user: CurrentUserDep,
    chats: ConversationRepoDep,
    request: Request,
) -> AskResponse:
    graph = getattr(request.app.state, "agentic_rag", None)
    if graph is None:
        raise RAGUnavailableError()

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

    langfuse = getattr(request.app.state, "langfuse", None)
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
    return AskResponse(
        answer=final.answer,
        citations=[Citation(article=c.article, quote=c.quote) for c in final.citations],
        conversation_id=chat_id,
    )
