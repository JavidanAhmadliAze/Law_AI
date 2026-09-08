import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from law_ai.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from law_ai.models.user import User


# Placeholder until the first question names the chat (routers/ask.py). Also the
# marker that a chat is still untitled, so ask.py knows it may overwrite it.
NEW_CHAT_TITLE = "New chat"


class Conversation(Base, UUIDMixin, TimestampMixin):
    """A chat thread. Its id doubles as the LangGraph thread id.

    The transcript lives inline in `messages` rather than in a child table:
    messages are only ever read as a whole conversation (GET /chats/{id}/messages),
    never queried on their own.

    Each entry is {"id", "role", "content", "created_at"}. SQLAlchemy cannot see
    in-place mutation of a JSONB list, so NEVER `chat.messages.append(...)` —
    ConversationRepository.add_message appends server-side with `||`, which is
    also what keeps concurrent appends from clobbering each other.
    """

    __tablename__ = "conversations"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), default=NEW_CHAT_TITLE, nullable=False)
    messages: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb"), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="conversations")
