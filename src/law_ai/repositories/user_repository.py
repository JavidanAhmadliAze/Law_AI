from pydantic import BaseModel
from sqlalchemy import select

from law_ai.models import User
from law_ai.repositories.base import BaseRepository


class _UserWrite(BaseModel):
    username: str
    password_hash: str


class UserRepository(BaseRepository[User, _UserWrite, _UserWrite]):
    model = User

    async def get_by_username(self, username: str) -> User | None:
        stmt = select(User).where(User.username == username)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
