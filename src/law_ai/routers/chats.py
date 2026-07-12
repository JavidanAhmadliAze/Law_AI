"""Chat management: list / create / delete + message history.

Ownership is enforced on every operation — a user only ever sees or deletes
their own conversations. Conversation id == LangGraph thread id, so deleting
a chat also removes its messages (DB cascade); checkpointer state cleanup is
wired in when the agent graph lands.
"""

import uuid

from fastapi import APIRouter, status

from law_ai.dependencies import ConversationRepoDep, CurrentUserDep
from law_ai.exceptions import NotFoundError
from law_ai.models import Conversation
from law_ai.schemas.chat import ChatCreate, ChatOut, MessageOut

router = APIRouter(prefix="/chats", tags=["chats"])


async def _owned_chat(
    chats: ConversationRepoDep, user_id: uuid.UUID, chat_id: uuid.UUID
) -> Conversation:
    chat = await chats.get(chat_id)
    if chat is None or chat.user_id != user_id:
        # 404 (not 403) so chat ids of other users are not probeable
        raise NotFoundError("Chat not found")
    return chat


@router.get("", response_model=list[ChatOut])
async def list_chats(user: CurrentUserDep, chats: ConversationRepoDep) -> list[ChatOut]:
    return [ChatOut.model_validate(c) for c in await chats.list_by_user(user.id)]


@router.post("", response_model=ChatOut, status_code=status.HTTP_201_CREATED)
async def create_chat(
    payload: ChatCreate, user: CurrentUserDep, chats: ConversationRepoDep
) -> ChatOut:
    chat = await chats.create({"user_id": user.id, "title": payload.title})
    return ChatOut.model_validate(chat)


@router.get("/{chat_id}/messages", response_model=list[MessageOut])
async def list_messages(
    chat_id: uuid.UUID, user: CurrentUserDep, chats: ConversationRepoDep
) -> list[MessageOut]:
    await _owned_chat(chats, user.id, chat_id)
    return [MessageOut.model_validate(m) for m in await chats.list_messages(chat_id)]


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: uuid.UUID, user: CurrentUserDep, chats: ConversationRepoDep) -> None:
    await _owned_chat(chats, user.id, chat_id)
    await chats.delete(chat_id)
