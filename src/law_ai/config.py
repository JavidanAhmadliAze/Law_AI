"""Application settings.

All service configuration is grouped into nested models and accumulated in a
single `Settings` class, loaded once via `dependencies.get_settings()`
(lru_cache). Env vars use "__" as the nesting delimiter, e.g. POSTGRES__HOST.
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    env: str = "local"  # local | staging | prod
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = "change-me"  # JWT signing key
    access_token_expire_minutes: int = 1440


class PostgresSettings(BaseModel):
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


class OpenSearchSettings(BaseModel):
    host: str = "localhost"
    port: int = 9200
    user: str = ""
    password: str = ""
    use_ssl: bool = False
    index: str = "law-chunks"
    search_pipeline: str = "hybrid-rrf"


class LLMSettings(BaseModel):
    """Provider and model are intentionally free-form — nothing is hardcoded."""

    provider: str = ""  # anthropic | bedrock | ...
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0


class EmbeddingSettings(BaseModel):
    model: str = ""
    provider: str = "local"  # local | api | bedrock
    api_url: str = ""
    dimension: int = 1024


class RerankerSettings(BaseModel):
    model: str = ""
    provider: str = "local"  # local | api | bedrock
    top_k: int = 5


class TranslationSettings(BaseModel):
    model: str = ""
    provider: str = "llm"  # glossary-only | model | llm


class S3Settings(BaseModel):
    bucket: str = ""
    region: str = "eu-central-1"
    endpoint_url: str = ""  # set for minio/localstack


class LangfuseSettings(BaseModel):
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    enabled: bool = False


class FetcherSettings(BaseModel):
    # act source URLs live in the registry (law_ai.acts), not in config
    data_dir: str = "data"


class Settings(BaseSettings):
    app: AppSettings = AppSettings()
    postgres: PostgresSettings = PostgresSettings()
    opensearch: OpenSearchSettings = OpenSearchSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    translation: TranslationSettings = TranslationSettings()
    s3: S3Settings = S3Settings()
    langfuse: LangfuseSettings = LangfuseSettings()
    fetcher: FetcherSettings = FetcherSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )
