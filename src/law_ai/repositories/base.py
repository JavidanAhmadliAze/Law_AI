"""Generic CRUD repository.

Concrete repositories inherit all CRUD and add only entity-specific queries.
Sessions are injected (owned by BaseDatabase.session()); repositories flush,
they NEVER commit — the session context manager owns the transaction.
"""

import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from law_ai.exceptions import NotFoundError
from law_ai.models.base import Base


class BaseRepository[ModelT: Base, CreateT: BaseModel, UpdateT: BaseModel]:
    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, entity_id: uuid.UUID) -> ModelT | None:
        return await self.session.get(self.model, entity_id)

    async def get_or_404(self, entity_id: uuid.UUID) -> ModelT:
        entity = await self.get(entity_id)
        if entity is None:
            raise NotFoundError(f"{self.model.__name__} not found")
        return entity

    async def list(self, *, limit: int = 100, offset: int = 0, **filters: Any) -> list[ModelT]:
        stmt = select(self.model).filter_by(**filters).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, data: CreateT | dict[str, Any]) -> ModelT:
        payload = data if isinstance(data, dict) else data.model_dump()
        entity = self.model(**payload)
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def update(self, entity_id: uuid.UUID, data: UpdateT | dict[str, Any]) -> ModelT:
        entity = await self.get_or_404(entity_id)
        payload = data if isinstance(data, dict) else data.model_dump(exclude_unset=True)
        for key, value in payload.items():
            setattr(entity, key, value)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity_id: uuid.UUID) -> None:
        entity = await self.get_or_404(entity_id)
        await self.session.delete(entity)
        await self.session.flush()

    async def count(self, **filters: Any) -> int:
        stmt = select(func.count()).select_from(self.model).filter_by(**filters)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def exists(self, **filters: Any) -> bool:
        stmt = select(self.model.id).filter_by(**filters).limit(1)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None
