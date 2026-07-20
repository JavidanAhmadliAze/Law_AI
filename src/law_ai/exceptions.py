"""Domain exceptions + FastAPI handlers.

Services and repositories raise these; the handlers (registered in main.py)
translate them to HTTP responses so routers stay free of error plumbing.

FastAPI is imported lazily: services raise these errors in FastAPI-free
environments too (Airflow tasks), where only the exception classes matter.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


class LawAIError(Exception):
    """Base class for all domain errors."""

    status_code = 500
    detail = "Internal server error"

    def __init__(self, detail: str | None = None) -> None:
        if detail is not None:
            self.detail = detail
        super().__init__(self.detail)


class NotFoundError(LawAIError):
    status_code = 404
    detail = "Resource not found"


class ConflictError(LawAIError):
    """e.g. username already taken."""

    status_code = 409
    detail = "Resource already exists"


class AuthenticationError(LawAIError):
    status_code = 401
    detail = "Invalid credentials"


class PermissionDeniedError(LawAIError):
    status_code = 403
    detail = "Not allowed"


class FetchError(LawAIError):
    """Upstream document source returned something other than the document
    (bot challenge, rate limit, outage) — retryable."""

    status_code = 502
    detail = "Upstream fetch failed"


class RetrievalError(LawAIError):
    detail = "Retrieval failed"


class GenerationError(LawAIError):
    detail = "Answer generation failed"


class TranslationError(LawAIError):
    detail = "Translation failed"


class GuardianBlockedError(LawAIError):
    """Question rejected by the guardian agent (off-topic or unsafe)."""

    status_code = 422
    detail = "Question was rejected"


class RAGUnavailableError(LawAIError):
    """Agent graph not built at boot (models unconfigured / a service down)."""

    status_code = 503
    detail = "RAG pipeline is not configured (set LLM__MODEL and EMBEDDING__MODEL)"


def register_exception_handlers(app: "FastAPI") -> None:
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.exception_handler(LawAIError)
    async def law_ai_error_handler(request: Request, exc: LawAIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
