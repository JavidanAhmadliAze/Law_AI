import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from law_ai.models.conversation import NEW_CHAT_TITLE


class ChatCreate(BaseModel):
    title: str = Field(default=NEW_CHAT_TITLE, max_length=255)


class ChatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    """One transcript entry, validated out of the conversation's JSONB list."""

    id: uuid.UUID
    role: str
    content: str
    created_at: datetime


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class AskResponse(BaseModel):
    answer: str
    conversation_id: uuid.UUID


class TokenResponse(BaseModel):
    """One streamed unit of the answer (SSE `token` event payload)."""

    text: str


class ErrorResponse(BaseModel):
    """Stream error payload (SSE `error` event)."""

    detail: str
