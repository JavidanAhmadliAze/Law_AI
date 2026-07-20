"""FastAPI dependency providers.

Settings are parsed and validated exactly once per process (lru_cache) and
shared by FastAPI, Gradio, Airflow tasks and CLI scripts alike. Long-lived
objects (database, services, agent graph) are created in the lifespan and
parked on app.state; providers here just hand them out.
"""

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from law_ai.config import Settings
from law_ai.database import BaseDatabase
from law_ai.exceptions import AuthenticationError, RAGUnavailableError
from law_ai.models import User
from law_ai.repositories.conversation_repository import ConversationRepository
from law_ai.repositories.user_repository import UserRepository
from law_ai.security import decode_access_token
from law_ai.services.cache.base import BaseCache
from law_ai.services.langfuse.client import LangfuseService


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_database(request: Request) -> BaseDatabase:
    return request.app.state.db


async def get_db_session(
    db: Annotated[BaseDatabase, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    async with db.session() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


# --- RAG stack (parked on app.state in the lifespan; optional at boot) ------


def get_agentic_rag(request: Request) -> Any:
    """The compiled agent graph — required; 503 when the RAG stack is disabled."""
    graph = getattr(request.app.state, "agentic_rag", None)
    if graph is None:
        raise RAGUnavailableError()
    return graph


def get_cache(request: Request) -> BaseCache | None:
    """Answer cache, or None when Redis was unavailable at boot."""
    return getattr(request.app.state, "cache", None)


def get_langfuse(request: Request) -> LangfuseService | None:
    """Tracer, or None when tracing is disabled."""
    return getattr(request.app.state, "langfuse", None)


AgentGraphDep = Annotated[Any, Depends(get_agentic_rag)]
CacheDep = Annotated[BaseCache | None, Depends(get_cache)]
LangfuseDep = Annotated[LangfuseService | None, Depends(get_langfuse)]


# --- repositories ---------------------------------------------------------


def get_user_repository(session: SessionDep) -> UserRepository:
    return UserRepository(session)


def get_conversation_repository(session: SessionDep) -> ConversationRepository:
    return ConversationRepository(session)


UserRepoDep = Annotated[UserRepository, Depends(get_user_repository)]
ConversationRepoDep = Annotated[ConversationRepository, Depends(get_conversation_repository)]


# --- auth ------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: SettingsDep,
    users: UserRepoDep,
) -> User:
    if credentials is None:
        raise AuthenticationError("Missing bearer token")
    user_id = decode_access_token(credentials.credentials, settings.app)
    user = await users.get(user_id)
    if user is None:
        raise AuthenticationError("User no longer exists")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]
