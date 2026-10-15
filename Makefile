.PHONY: install lint format typecheck test test-unit test-integration \
        up down logs stop-app ingest-up ingest-down ingest-logs stop-ingest \
        migrate migration run dev eval eval-retrieval eval-generation clean

# ---------------------------------------------------------------- setup ----

install:            ## Install all deps (incl. dev) into .venv
	uv sync --group dev

install-ml:         ## Also install local embedding/reranker deps (torch — heavy)
	uv sync --group dev --group ml

# ---------------------------------------------------------------- quality ----

lint:               ## Ruff lint + format check
	uv run ruff check src tests
	uv run ruff format --check src tests

format:             ## Auto-fix lint + format
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:          ## Mypy (gradio import first: it generates its .pyi stubs on first import)
	uv run python -c "import gradio"
	uv run mypy

# ---------------------------------------------------------------- tests ----

test: test-unit     ## Default: unit tests

test-unit:
	uv run pytest tests/unit -q

test-integration:   ## Requires `make up` first
	uv run pytest tests/integration -q -m integration

# ---------------------------------------------------------------- infra ----
#
# Two stacks, one shared infrastructure layer (docker-compose.infra.yml):
#   app     = postgres + opensearch + tei + redis + app
#   ingest  = postgres + opensearch + tei + airflow
# Both run under project `law-ai`, so the shared containers are the SAME ones.
# `down` therefore stops infrastructure the other stack may still be using;
# use `stop-app` / `stop-ingest` to stop only that stack's own service.

APP     := docker compose -f docker/docker-compose.yml
INGEST  := docker compose -f docker/docker-compose.ingest.yml


build:              ## Build the app container (no-op if already built)
	$(APP) build

up:                 ## Start the application stack (postgres, opensearch, tei, redis, app)
	$(APP) up -d

down:               ## Stop the application stack AND the shared infrastructure
	$(APP) down

logs:
	$(APP) logs -f

stop-app:           ## Stop only the app container, leave infrastructure up
	$(APP) stop app

ingest-up:          ## Start the ingestion stack (postgres, opensearch, tei, airflow) — UI on :8080
	$(INGEST) up -d

ingest-down:        ## Stop the ingestion stack AND the shared infrastructure
	$(INGEST) down

ingest-logs:
	$(INGEST) logs -f airflow

stop-ingest:        ## Stop only Airflow, leave infrastructure up for the app
	$(INGEST) stop airflow

# ---------------------------------------------------------------- db ----

migrate:            ## Apply migrations
	uv run alembic upgrade head

migration:          ## Autogenerate a migration: make migration m="add users"
	uv run alembic revision --autogenerate -m "$(m)"

# ---------------------------------------------------------------- run ----

dev:                ## Run API + Gradio with reload
	uv run uvicorn law_ai.main:app --reload --host 0.0.0.0 --port 8000

run:
	uv run uvicorn law_ai.main:app --host 0.0.0.0 --port 8000

# ---------------------------------------------------------------- eval ----

eval: eval-retrieval eval-generation

eval-retrieval:     ## Deterministic, cheap — run often
	uv run python -m evaluation.runners.run_eval --mode retrieval

eval-generation:    ## LLM-as-judge — costs tokens, run on PRs/releases
	uv run python -m evaluation.runners.run_eval --mode generation

eval-ui:            ## Render evaluation reports into reports/eval_dashboard.html
	uv run python -m evaluation.report_ui

# ---------------------------------------------------------------- misc ----

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

help:               ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
