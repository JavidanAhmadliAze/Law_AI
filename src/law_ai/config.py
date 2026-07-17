"""Application settings.

Each settings block reads its own environment section via an env prefix
(e.g. APP__, POSTGRES__) — all inherit the .env location and parsing rules
from `BaseConfigSettings`. The root `Settings` aggregates them through
`Field(default_factory=...)`, so every `Settings()` call re-reads the current
environment (class-definition-time defaults would go stale). Loaded once per
process via `dependencies.get_settings()` (lru_cache).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# absolute path so .env is found regardless of the process's working
# directory (config.py lives at src/law_ai/config.py → repo root is 2 up);
# missing file (e.g. inside containers) is silently skipped — real env
# variables still apply
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class BaseConfigSettings(BaseSettings):
    """Shared env-parsing rules; subclasses add their own `env_prefix`.

    pydantic merges `model_config` across inheritance, so subclasses only
    declare what differs (their prefix).
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        extra="ignore",
        case_sensitive=False,
    )


class AppSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="APP__")

    env: str = "local"  # local | staging | prod
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = ""  # JWT signing key
    access_token_expire_minutes: int = 1440


class PostgresSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES__")

    host: str = "localhost"
    port: int = 5432
    user: str = "lawai"
    password: str = "lawai"
    db: str = "lawai"
    pool_size: int = 10

    @property
    def async_dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def sync_dsn(self) -> str:
        """Used by Alembic and the LangGraph checkpointer setup."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class OpenSearchSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="OPENSEARCH__")

    host: str = "localhost"
    port: int = 9200
    user: str = ""
    password: str = ""
    use_ssl: bool = False
    index: str = "law-chunks"
    search_pipeline: str = "hybrid-rrf"


class LLMSettings(BaseConfigSettings):
    """Provider and model are intentionally free-form — nothing is hardcoded."""

    model_config = SettingsConfigDict(env_prefix="LLM__")

    provider: str = ""  # anthropic | bedrock | ...
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0


class EmbeddingSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING__")

    model: str = ""
    provider: str = "local"  # local | api | bedrock
    api_url: str = ""
    dimension: int = 1024
    batch_size: int = 32  # texts per embed request (TEI rejects oversized batches)
    timeout_seconds: float = 300.0  # generous: CPU inference on big batches is slow


class RerankerSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKER__")

    model: str = ""
    provider: str = "local"  # local | api | bedrock
    top_k: int = 5


class TranslationSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="TRANSLATION__")

    model: str = ""
    provider: str = "llm"  # glossary-only | model | llm


class S3Settings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="S3__")

    bucket: str = ""
    region: str = "eu-central-1"
    endpoint_url: str = ""  # set for minio/localstack


class LangfuseSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="LANGFUSE__")

    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    enabled: bool = False


class FetcherSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="FETCHER__")

    # act source URLs live in the registry (law_ai.acts), not in config
    data_dir: str = "data"


class RedisSettings(BaseConfigSettings):
    """Exact-match RAG answer cache (services/cache)."""

    model_config = SettingsConfigDict(env_prefix="REDIS__")

    enabled: bool = True  # app degrades gracefully when Redis is unreachable
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    ttl_hours: int = 24
    socket_timeout_seconds: float = 2.0
    socket_connect_timeout_seconds: float = 2.0


class Settings(BaseConfigSettings):
    """Aggregates all sections; env parsing rules inherited from the base.

    The root needs no env_prefix — its fields are whole sections, each
    populated by its own factory reading its own prefixed variables.
    """

    app: AppSettings = Field(default_factory=AppSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    fetcher: FetcherSettings = Field(default_factory=FetcherSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
