# Law-AI — Architecture

Agentic RAG over the Polish Constitution. Users ask in English or Polish;
retrieval runs over the Polish legal corpus; answers come back grounded and
cited.

Two independent pipelines:

- **Offline (Airflow)** — builds the index:
  `fetching → pdf_parsing → chunking → metadata_creating → embedding → indexing → storing`
- **Online (`services/agents`)** — serves queries with a hybrid adaptive agent
  graph (fast path for simple questions, supervisor + sub-agents for complex).

---

## 1. Root layout

```
Law_AI/
├── airflow/               # offline orchestration only — DAGs stay thin
├── src/law_ai/            # application package (§2)
├── tests/                 # unit / integration / fixtures
├── evaluation/            # RAG eval: retrieval + generation (§7)
├── migrations/            # Alembic (Postgres)
├── docker/                # Dockerfile, Dockerfile.airflow, docker-compose.yml
├── scripts/               # one-off helpers
├── .env.example  Makefile  pyproject.toml (uv)  .pre-commit-config.yaml
└── .github/workflows/ci.yml
```

Stores: **OpenSearch** (hybrid dense+sparse index + reranking), **Postgres**
(users, conversations, LangGraph checkpoints), **S3** (raw/parsed PDFs,
optional locally). Every external capability sits behind an
**interface + factory**, so self-hosted ↔ AWS-managed (RDS, Amazon OpenSearch,
Bedrock, MWAA) is a config swap, never a rewrite.

## 2. `src/law_ai/` — strict layering

```
routers → services → repositories → database
```

Routers never touch the DB. Repositories hold no business logic. All AI logic
lives in `services/`. Airflow imports `services/*` without importing FastAPI.

```
src/law_ai/
├── main.py           # app factory + lifespan: initializes DATABASE and all SERVICES
├── config.py         # nested pydantic-settings; env delimiter "__"
├── dependencies.py   # @lru_cache get_settings, get_current_user, providers
├── exceptions.py  middleware.py  logging.py
├── gradio_app.py     # login → chat sidebar → thread view (mounted on FastAPI)
├── database/         # §3
├── models/           # User, Conversation, Message
├── schemas/          # pydantic request/response
├── repositories/     # §4 — Postgres CRUD only
├── routers/          # auth, chats, ask, health
├── clients/          # shared low-level singletons
└── services/         # §5
```

**Lifespan contract (`main.py`):** on startup, create the database via
`DatabaseFactory` (`await db.startup()`), create every service via its factory,
inject services into `AgentGraphFactory`, park everything on `app.state`. On
shutdown, `await db.teardown()` and close service clients.

## 3. `database/` — interface + factory

```
database/
├── factory.py                 # DatabaseFactory.create(settings) → BaseDatabase
└── interface/
    ├── base.py                # BaseDatabase(ABC)
    └── postgres.py            # PostgresDatabase
```

`BaseDatabase` abstract methods: `startup()` (engine+pool, fail fast),
`teardown()` (dispose), `session()` (async unit-of-work CM — commits/rolls
back), `health_check()` (`SELECT 1`).

Purpose: **users and their conversations** (plus LangGraph checkpointer state —
same Postgres). One instance, created in lifespan, on `app.state.db`.
Repositories receive sessions and never call `commit()`.

## 4. `repositories/`

`BaseRepository[Model, CreateSchema, UpdateSchema]` implements all CRUD:
`get, get_or_404, list, create, update, delete, count, exists`. Concrete repos
(`UserRepository`, `ConversationRepository`) add only entity-specific queries
(`get_by_username`, `list_by_user`). Postgres only — OpenSearch is a service.

## 5. `services/` — isolated capabilities, factory pattern

Each service: `base.py` (interface) · `client.py` (implementation) ·
`factory.py`. Services do not know about each other's internals; consumers
receive them by injection.

| Service | Owns |
|---|---|
| `fetcher/` | downloading constitution PDFs |
| `pdf_parser/` | PDF → structured text (articles/sections) |
| `chunking/` | article-aware splitting |
| `metadata/` | article no., chapter, title, source URL, dates, language |
| `embedding/` | dense (+ optional sparse) vectors — multilingual/Polish model |
| `opensearch/` | **indexing + hybrid search (dense kNN + sparse, RRF fusion) + reranking + metadata filtering** — all retrieval mechanics live here |
| `llm/` | chat completions; provider + model are env-driven, never hardcoded |
| `translation/` | EN↔PL: glossary (regex, deterministic legal terms) → NMT model → LLM escalation; `detect_language` short-circuit |
| `langfuse/` | tracing callback |
| `agents/` | the **agentic_rag pipeline** — consumes the services above, creates none of them |

Notes:
- Translation matters most for the **sparse leg**: BM25 is lexical, so the
  glossary feeds canonical Polish legal terms (`freedom of speech → wolność
  słowa`). Dense retrieval with a multilingual embedder bridges EN↔PL on its
  own.
- Quoted constitutional text used for citations is **never translated** —
  verbatim Polish + article id travel through state.

## 6. `services/agents/` — the online pipeline

Agents are: **guardian, query_rewriter, supervisor, sub_agents, writer.**

```
agents/
├── agentic_rag.py    # builds the StateGraph (fast + complex paths)
├── factory.py        # AgentGraphFactory.create(opensearch, translation, llm,
│                     #   checkpointer, tracer) → compiled graph
├── state.py          # GraphState — carries the conversation between agents;
│                     #   parallel results merge via additive reducers
├── tools.py          # think_tool + compress_tool (reasoning tools ONLY)
├── prompts.py        # per-agent system prompts
├── schemas.py        # structured output per LLM call → deterministic routing
└── nodes/
    ├── guardian.py       # security (injection/unsafe) + relevance gate → typed verdict
    ├── query_rewriter.py # rewrite/decompose + metadata filters + route simple|complex
    ├── supervisor.py     # fan-out sub_agents (Send API), think_tool coverage loop
    ├── sub_agent.py      # per sub-question: opensearch.retrieve + translation
    └── writer.py         # llm.generate → grounded, cited answer
```

**Flow (adaptive):**

```
guardian → query_rewriter ─route─┐
  simple  → translate(EN→PL) → opensearch.retrieve(hybrid+filter+rerank)
            → translate(PL→EN context) ───────────────────────────┐
  complex → supervisor ⇄ sub_agents (translate → retrieve → think │
            → compress), supervisor think_tool loop ──────────────┤
                                                        → writer → END
```

**Service call vs tool (the rule):** retrieval, translation, and generation are
**injected service calls** inside nodes — not LLM tools. Only two tools exist:
`think_tool` (self-reflection: "is the evidence sufficient?" — drives sub_agent
retries and the supervisor's final go/no-go) and `compress_tool` (context
engineering: distill evidence into `{claim, source_article, quote}` records,
citations verbatim, so GraphState never accumulates raw passage dumps).

Memory: short-term = LangGraph **Postgres checkpointer** (chat id = thread id);
long-term optional via LangGraph Store. Langfuse is attached as a callback on
the compiled graph — every node/LLM call is traced.

## 7. `evaluation/` — retrieval and generation, separately

Retrieval and generation fail independently, so they are measured separately.

- **retrieval/** — deterministic, no LLM: recall@k, precision@k, MRR, nDCG,
  hit-rate; reported **per leg** (dense / sparse / fused) to justify hybrid and
  tune fusion.
- **generation/** — LLM-as-judge (RAGAS behind our own metrics interface):
  faithfulness, answer relevance, correctness, **citation accuracy** (article
  numbers must match sources — critical for a legal system).
- `datasets/golden_qa.jsonl` — question → expected article ids + reference
  answer; includes **cross-lingual** (EN question → PL article) cases.
- Judge model ≠ answer model; pinned in `evaluation/config.py`, env-driven.
- `make eval-retrieval` (cheap, run often) / `make eval-generation` (on PRs).

## 8. Offline pipeline (Airflow)

```
airflow/dags/ingest_constitution.py
fetching → pdf_parsing → chunking → metadata_creating → embedding → indexing → storing
```

DAG tasks are thin wrappers over `services/*`. `metadata_creating` produces the
fields online **metadata filtering** relies on, so it runs at ingest. Writes
are idempotent (re-runs never duplicate). Storing targets: OpenSearch (index),
Postgres (doc metadata if needed), S3 (source of truth for PDFs).

## 9. Auth + chats

- `User(username UNIQUE, password_hash)` — bcrypt, JWT via `APP__SECRET_KEY`.
- `Conversation(user_id, title)` + `Message(role, content)`; conversation id ==
  LangGraph thread id. Deleting a chat cascades messages + checkpoint state.
- Routers: `auth` (register/login), `chats` (list/create/delete),
  `ask` (`POST /chats/{id}/ask`), `health`.
- Gradio: login screen → left sidebar lists chats (select / new / delete) →
  right pane thread + input.

## 10. Model choices (env-driven, nothing hardcoded)

Starting point: **BGE-M3** embeddings (dense+sparse dual output → maps 1:1 to
the hybrid index) + **bge-reranker-v2-m3**. Benchmark against the
Polish-specific pair (**mmlw-retrieval-roberta-large-v2** +
**polish-reranker-large-ranknet**) with `evaluation/retrieval` — the winner is
an env swap. AWS path: Cohere embed-multilingual + Cohere Rerank via Bedrock.
LLM provider/model: any, set via `LLM__PROVIDER` / `LLM__MODEL`.
