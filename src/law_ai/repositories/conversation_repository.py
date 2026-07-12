import uuid

from pydantic import BaseModel
from sqlalchemy import select

from law_ai.models import Conversation, Message
from law_ai.repositories.base import BaseRepository


class _ConversationWrite(BaseModel):
    user_id: uuid.UUID | None = None
    title: str | None = None


class ConversationRepository(BaseRepository[Conversation, _ConversationWrite, _ConversationWrite]):
    model = Conversation

    async def list_by_user(self, user_id: uuid.UUID) -> list[Conversation]:
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_messages(self, conversation_id: uuid.UUID) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def add_message(self, conversation_id: uuid.UUID, role: str, content: str) -> Message:
        message = Message(conversation_id=conversation_id, role=role, content=content)
        self.session.add(message)
        await self.session.flush()
        await self.session.refresh(message)
        return message
