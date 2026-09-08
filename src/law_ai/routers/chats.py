"""Chat management: list / create / delete + message history.

Ownership is enforced on every operation — a user only ever sees or deletes
their own conversations. Conversation id == LangGraph thread id, so deleting
a chat also removes its messages (DB cascade); checkpointer state cleanup is
wired in when the agent graph lands.
"""

import uuid

from fastapi import APIRouter, status

from law_ai.dependencies import ConversationRepoDep, CurrentUserDep
from law_ai.schemas.chat import ChatCreate, ChatOut, MessageOut

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("", response_model=list[ChatOut])
async def list_chats(user: CurrentUserDep, chats: ConversationRepoDep) -> list[ChatOut]:
    return [ChatOut.model_validate(c) for c in await chats.list_by_user(user.id)]


@router.post("", response_model=ChatOut, status_code=status.HTTP_201_CREATED)
async def create_chat(
    payload: ChatCreate, user: CurrentUserDep, chats: ConversationRepoDep
) -> ChatOut:
    # an unused chat from an earlier click has no messages and no real title —
    # clear those out here rather than let them accumulate
    await chats.delete_empty(user.id)
    chat = await chats.create({"user_id": user.id, "title": payload.title})
    return ChatOut.model_validate(chat)


@router.get("/{chat_id}", response_model=ChatOut)
async def get_chat(chat_id: uuid.UUID, user: CurrentUserDep, chats: ConversationRepoDep) -> ChatOut:
    """Fetch one conversation. 404 when it does not exist *or* is not the
    caller's — see ConversationRepository.get_chat."""
    return ChatOut.model_validate(await chats.get_chat(chat_id, user.id))


@router.get("/{chat_id}/messages", response_model=list[MessageOut])
async def list_messages(
    chat_id: uuid.UUID, user: CurrentUserDep, chats: ConversationRepoDep
) -> list[MessageOut]:
    await chats.get_chat(chat_id, user.id)  # 404s unless the caller owns it
    return [MessageOut.model_validate(m) for m in await chats.list_messages(chat_id)]


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: uuid.UUID, user: CurrentUserDep, chats: ConversationRepoDep) -> None:
    await chats.get_chat(chat_id, user.id)  # 404s unless the caller owns it
    await chats.delete(chat_id)
