import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from law_ai.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from law_ai.models.message import Message
    from law_ai.models.user import User


class Conversation(Base, UUIDMixin, TimestampMixin):
    """A chat thread. Its id doubles as the LangGraph thread id."""

    __tablename__ = "conversations"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), default="New chat", nullable=False)

    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
