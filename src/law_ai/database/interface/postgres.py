"""PostgreSQL implementation of BaseDatabase (SQLAlchemy async + asyncpg)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from law_ai.config import PostgresSettings
from law_ai.database.interface.base import BaseDatabase
from law_ai.logging import get_logger

logger = get_logger(__name__)


class PostgresDatabase(BaseDatabase):
    def __init__(self, settings: PostgresSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def startup(self) -> None:
        self._engine = create_async_engine(
            self._settings.async_dsn,
            pool_size=self._settings.pool_size,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, autoflush=False
        )
        # fail fast if unreachable
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("database.startup", host=self._settings.host, db=self._settings.db)

    async def teardown(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("database.teardown")

    @asynccontextmanager
    async def _session_cm(self) -> AsyncIterator[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("Database not started — call startup() first")
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def session(self):  # type: ignore[no-untyped-def]  # AbstractAsyncContextManager[AsyncSession]
        return self._session_cm()

    async def health_check(self) -> bool:
        if self._engine is None:
            return False
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.warning("database.health_check_failed", exc_info=True)
            return False
