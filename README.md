# Law-AI

Agentic RAG over the Polish Constitution — ask legal questions in English or
Polish, get grounded answers with citations to the original articles.

- **Offline**: Airflow pipeline fetches/parses/chunks the constitution and
  builds a hybrid (dense + sparse) OpenSearch index.
- **Online**: a LangGraph agent pipeline (guardian → query rewriter →
  supervisor/sub-agents → writer) retrieves, reranks, and generates cited
  answers. FastAPI + Gradio chat UI with auth and per-user chat history.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Stack

FastAPI · Gradio · LangGraph · OpenSearch (hybrid + rerank) · Postgres ·
Airflow · Langfuse · uv · Docker

## Quickstart

```bash
# 1. deps
make install                 # uv sync (add `make install-ml` for local models)

# 2. env
cp .env.example .env         # fill in LLM__*, EMBEDDING__*, ...

# 3. infra (postgres + opensearch + airflow)
make up

# 4. db schema
make migrate

# 5. run API + Gradio
make dev                     # http://localhost:8000  (Gradio at /ui)
```

Airflow UI: http://localhost:8080 (admin/admin) — trigger `ingest_constitution`
to build the index.

## Development

```bash
make lint         # ruff check + format check
make format       # auto-fix
make typecheck    # mypy
make test         # unit tests
make test-integration   # needs `make up`
make eval-retrieval     # RAG retrieval metrics (cheap, deterministic)
make eval-generation    # LLM-as-judge (costs tokens)
```

Pre-commit: `uv run pre-commit install`
