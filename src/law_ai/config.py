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
    def standard_dsn(self) -> str:
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
    act_name_boost: float = 0.0
    candidates_per_leg: int = 50
    union_per_leg: int = 20
    # How the two legs become one candidate pool:
    #   union  each leg's top `union_per_leg`, no fusion score; the cross-encoder
    #          decides among them (the production design).
    #   rrf    reciprocal-rank fusion over each leg's full `candidates_per_leg`,
    #          cut to `top_k` by fused score — the earlier design, kept so the two
    #          can be measured against each other on the same queries.
    fusion: str = "union"  # union | rrf
    rrf_k: int = 60  # the RRF constant: score = sum over legs of 1 / (rrf_k + rank)
    # Which legs run: hybrid (dense + bm25) is production; dense / bm25 alone
    # exist to measure each leg's contribution.
    legs: str = "hybrid"  # hybrid | dense | bm25
    """How many hits each leg contributes to the candidate union.

    The candidate set is the union of the dense leg's top `union_per_leg` and
    BM25's top `union_per_leg` — at most 20 chunks, fewer wherever the legs agree.

    Lowered from 15. The delivered set is capped at `agent.max_chunks`, so every
    extra candidate a sub-query contributes competes for the same slots: a leg
    depth of 15 per sub-query put 150 candidates in front of 30 slots on a
    six-search plan, and the provisions that fell out were the ones a weaker
    sub-query had found. A shallower union also halves the cross-encoder's work,
    which is the run's cost on this machine.

    This replaced reciprocal rank fusion. RRF scored a chunk one leg found at
    ~1/(k+r) and a chunk both legs found mediocre at ~2/(k+r), so a strong
    single-leg hit lost to two weak dual-leg ones: a provision at dense rank 1
    reached the reranker at fused rank 16. A union has no such interaction —
    each leg's best survive on their own merits and the cross-encoder, which can
    read the provision, decides between them.

    `candidates_per_leg` above is still what each leg FETCHES; this is how much
    of it enters the union.
    """


class LLMSettings(BaseConfigSettings):
    """Provider and model are intentionally free-form — nothing is hardcoded."""

    model_config = SettingsConfigDict(env_prefix="LLM__")

    provider: str = ""
    model: str = ""
    fast_model: str = ""
    reasoning_model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    structured_method: str = "function_calling"
    reasoning_effort: str = ""
    """Thinking budget for the reasoning model — "low" | "high" | "max", provider-defined.

    Carried on the same settings object as the model id because it belongs to
    the model, not to the caller: DeepSeek-V4.1-Flash is a reasoning model only
    when it is asked to be one, and a node that wants reasoning wants the budget
    too. Only `create_reasoning_llm` passes it on; the generation and fast paths
    clear it, so a model that does not support the parameter never sees it.
    """


class EmbeddingSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING__")

    model: str = ""
    provider: str = "local"  # local | api
    api_url: str = ""
    dimension: int = 1024
    batch_size: int = 32  # texts per embed request (TEI rejects oversized batches)
    timeout_seconds: float = 300.0  # generous: CPU inference on big batches is slow


class RerankerSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKER__")

    model: str = ""  # in-process CrossEncoder; empty disables reranking (RRF-only)
    batch_size: int = 4
    device: str = ""  # "" lets sentence-transformers choose; "cpu" / "cuda" to force
    fp16: bool = False
    """Load the cross-encoder in half precision — the weights, not the maths.

    A 1024-hidden 24-layer cross-encoder is ~1.7 GB of fp32 weights, and loading
    it alongside TEI, OpenSearch and an editor on an 8 GB laptop drove the
    machine into a global OOM twice: swap 3.6/4.0 GB, and the kernel killed the
    editor while the run died with it. fp16 halves the resident weights.

    Off by default, and MEASURED USELESS on this CPU — do not reach for it
    again without re-measuring. 32 pairs through this cross-encoder:

        fp32   load 11.2s   score  9.7s   peak RSS 2.1 GB
        fp16   load  7.0s   score 66.4s   peak RSS 2.8 GB

    6.8x slower AND more resident, because CPU fp16 has no native kernels for
    most ops: torch upcasts per operation and allocates fp32 buffers anyway, so
    the half-precision weights buy nothing and the conversions cost everything.
    It would only pay on a GPU.
    """


class AgentSettings(BaseConfigSettings):
    """Tunables for the retrieval chain — never hardcoded in the nodes."""

    model_config = SettingsConfigDict(env_prefix="AGENT__")
    score_floor: float = 4.0
    """Denominator floor when normalising a sub-query's rerank scores.

    Each sub-query's chunks are divided by `max(its best score, score_floor)`
    before the merge, so the sort compares chunks against their own sub-query's
    best rather than across incomparable scales.

    Without the floor, a sub-query whose best hit scores 1.2 would have that
    chunk rescaled to 1.0 and placed level with another sub-query's genuine top
    hit. The floor means only a sub-query that actually found something strong
    gets to normalise against itself.
    """
    max_chunks: int = 30
    # Delivery cut applied after the merge (0 = off: deliver the top `max_chunks`).
    # Keeps chunks whose normalised rerank score is >= delivery_score_cut, but
    # never fewer than max(delivery_min_chunks, number of sub-queries). Measured
    # on the showcase set: 0.85 with floor 2 lifts precision 0.09 -> 0.80 at a
    # recall cost of ~9 points (0.977 -> 0.886); 0.5 keeps recall intact at
    # precision 0.50. The writer then reads ~3 provisions instead of 30.
    delivery_score_cut: float = 0.0
    delivery_min_chunks: int = 2
    decompose_queries: bool = True
    rerank_candidates: int = 50
    max_searches: int = 6
    # Optional: a JSONL of lawyer-written {sub-query: act} labels (rows with a
    # "label" dict, e.g. dataset_corrected.jsonl). Shown to the planner grouped
    # by act so a recurring provision gets the lawyer's wording; measured +9-11
    # points of recall on held-out questions. Empty disables it.
    label_bank_path: str = ""


class LangfuseSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="LANGFUSE__")

    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    enabled: bool = False


class FetcherSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(env_prefix="FETCHER__")

    # act source URLs live in the registry (law_ai.acts), not in config


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
    agent: AgentSettings = Field(default_factory=AgentSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    fetcher: FetcherSettings = Field(default_factory=FetcherSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
