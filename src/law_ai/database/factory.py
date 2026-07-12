"""Database factory — call sites receive BaseDatabase, never a concrete class."""

from law_ai.config import Settings
from law_ai.database.interface.base import BaseDatabase
from law_ai.database.interface.postgres import PostgresDatabase


class DatabaseFactory:
    @staticmethod
    def create(settings: Settings) -> BaseDatabase:
        # Postgres is the only backend today; adding another (e.g. a test
        # double) is a new branch here — call sites stay untouched.
        return PostgresDatabase(settings.postgres)
