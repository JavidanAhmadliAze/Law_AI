"""Application entrypoint.

The lifespan initializes the database and all services via their factories
and parks them on app.state; shutdown tears everything down in reverse.

The RAG stack (llm/embedding/opensearch/translation/agents) is optional at
boot: with incomplete model config the API still serves auth/chats/health,
and /ask answers 503 until the models are configured.
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from law_ai.database import create_database
from law_ai.dependencies import get_settings
from law_ai.exceptions import register_exception_handlers
from law_ai.logging import get_logger, setup_logging
from law_ai.middleware import register_middleware
from law_ai.routers import ask, auth, chats, health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan for RAG API: initialize database, services, agent graph; teardown in reverse."""

    logger.info("Starting RAG API")

    settings = get_settings()
    setup_logging(settings)
    stack = AsyncExitStack()

    # --- database (users + conversations) --------------------------------
    db = create_database(settings)
    await db.startup()
    app.state.db = db
    logger.info("Database is connected")

    # --- answer cache (optional — misses are just slower, never fatal) ----
    app.state.cache = None
    try:
        from law_ai.services.cache.factory import create_cache_client

        cache = create_cache_client(settings)
        await cache.startup()
        app.state.cache = cache
    except Exception as exc:  # noqa: BLE001 — degraded boot is intentional
        logger.warning("cache.disabled", reason=str(exc))

    # --- services → agent graph ------------------------------------------
    app.state.agentic_rag = None
    app.state.langfuse = None
    app.state.search = None
    try:
        from law_ai.services.agents.factory import create_agent_graph
        from law_ai.services.embedding.factory import create_embedder
        from law_ai.services.langfuse.factory import create_langfuse
        from law_ai.services.llm.factory import create_llm
        from law_ai.services.opensearch.factory import create_search_service
        from law_ai.services.translation.factory import create_translator

        llm = create_llm(settings)
        embedder = create_embedder(settings)
        app.state.langfuse = create_langfuse(settings)
        search = create_search_service(settings, embedder, tracer=app.state.langfuse)
        await search.startup()
        app.state.search = search
        translator = create_translator(settings, llm=llm)

        checkpointer = None
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(settings.postgres.sync_dsn)
            )
            await checkpointer.setup()
        except Exception as exc:  # graph still works, just without thread memory
            logger.warning("checkpointer.unavailable", error=str(exc))

        app.state.agentic_rag = create_agent_graph(
            llm=llm,
            search=search,
            translator=translator,
            checkpointer=checkpointer,
        )
        logger.info("rag.ready")
    except (ValueError, Exception) as exc:  # noqa: BLE001 — degraded boot is intentional
        logger.warning("rag.disabled", reason=str(exc))

    logger.info("app.startup", env=settings.app.env)
    yield

    if app.state.cache is not None:
        await app.state.cache.teardown()
    if app.state.search is not None:
        await app.state.search.teardown()
    if app.state.langfuse is not None:
        app.state.langfuse.shutdown()
    await stack.aclose()
    await db.teardown()
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Law-AI",
        description="Agentic RAG over Polish law",
        version="0.1.0",
        lifespan=lifespan,
    )
    register_middleware(app)
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(chats.router)
    app.include_router(ask.router)

    from law_ai.gradio_app import mount_gradio

    mount_gradio(app)  # chat UI at /ui

    return app


app = create_app()
