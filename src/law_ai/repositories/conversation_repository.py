import json
import uuid
from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy import CursorResult, delete, func, literal, select, update
from sqlalchemy import cast as sa_cast
from sqlalchemy.dialects.postgresql import JSONB

from law_ai.exceptions import NotFoundError
from law_ai.models import Conversation
from law_ai.models.base import utcnow
from law_ai.repositories.base import BaseRepository


class _ConversationWrite(BaseModel):
    user_id: uuid.UUID | None = None
    title: str | None = None


class ConversationRepository(BaseRepository[Conversation, _ConversationWrite, _ConversationWrite]):
    model = Conversation

    async def get_chat(self, chat_id: uuid.UUID, user_id: uuid.UUID) -> Conversation:
        """Owner-scoped fetch — the authorization gate for every per-chat route.

        Ownership is a WHERE clause, not a post-fetch comparison: another user's
        row is never selected in the first place. Missing and not-yours both
        raise 404 (never 403), so foreign chat ids stay unprobeable. This matters
        more than it looks — conversation id == LangGraph thread id, so a leak
        here would let a user stream into someone else's checkpointed thread.
        """
        stmt = select(Conversation).where(
            Conversation.id == chat_id, Conversation.user_id == user_id
        )
        result = await self.session.execute(stmt)
        chat = result.scalar_one_or_none()
        if chat is None:
            raise NotFoundError("Chat not found")
        return chat

    async def delete_empty(self, user_id: uuid.UUID) -> int:
        """Drop the caller's conversations that never received a message.

        The UI creates a row the moment "New chat" is clicked, so a user who
        opens a chat and then never asks anything leaves a dead row behind — and
        since titles are only assigned on the first question (routers/ask.py),
        an empty chat is an untitled one too: nothing to show, nothing to
        resume. Sweeping on the next create bounds these to at most one per
        user without needing a scheduled job.

        Returns the number of rows removed.
        """
        stmt = (
            delete(Conversation)
            .where(
                Conversation.user_id == user_id,
                func.jsonb_array_length(Conversation.messages) == 0,
            )
            .execution_options(synchronize_session=False)
        )
        # execute() is typed Result; a DELETE always yields a CursorResult
        result = cast("CursorResult[Any]", await self.session.execute(stmt))
        return int(result.rowcount)

    async def list_by_user(self, user_id: uuid.UUID) -> list[Conversation]:
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_messages(self, conversation_id: uuid.UUID) -> list[dict[str, Any]]:
        """The conversation's transcript, oldest first (append order)."""
        stmt = select(Conversation.messages).where(Conversation.id == conversation_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() or []

    async def add_message(
        self, conversation_id: uuid.UUID, role: str, content: str
    ) -> dict[str, Any]:
        """Append one message to the conversation's transcript.

        The append happens in the database (`messages || '[...]'::jsonb`), not
        by loading the list and rewriting it: two requests appending to the same
        chat concurrently would otherwise read the same array and the later
        write would silently drop the earlier message.
        """
        entry: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "created_at": utcnow().isoformat(),
        }
        stmt = (
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(messages=Conversation.messages + sa_cast(literal(json.dumps([entry])), JSONB))
            .execution_options(synchronize_session=False)
        )
        await self.session.execute(stmt)
        return entry
