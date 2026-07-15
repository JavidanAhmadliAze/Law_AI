"""Abstract database contract.

Everything downstream (dependencies, repositories) depends on this interface,
never on a concrete implementation — swapping Postgres for RDS (or a test
double) is a factory/config change only.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager

from sqlalchemy.ext.asyncio import AsyncSession


class BaseDatabase(ABC):
    """Lifecycle + session provider for the relational store (users, chats)."""

    @abstractmethod
    async def startup(self) -> None:
        """Create engine + connection pool and verify connectivity.

        Called once from the app lifespan; must fail fast if the database is
        unreachable.
        """

    @abstractmethod
    async def teardown(self) -> None:
        """Dispose the engine and drain the pool. Called once on shutdown."""

    @abstractmethod
    def session(self) -> AbstractAsyncContextManager[AsyncSession]:
        """Yield a unit-of-work session.

        Commits on success, rolls back on error. This is the ONLY way
        repositories obtain sessions; repositories never call commit().
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Cheap connectivity probe (SELECT 1) for /health."""

    # Helper for FastAPI dependencies ------------------------------------
    async def iter_session(self) -> AsyncIterator[AsyncSession]:
        """Adapter so `session()` can be used as a FastAPI dependency."""
        async with self.session() as session:
            yield session
