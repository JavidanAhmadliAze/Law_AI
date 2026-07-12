"""Langfuse tracing — attached to the agent graph as a LangChain callback.

When disabled (LANGFUSE__ENABLED=false) the service returns no handler and
the graph runs untraced; call sites never branch on config themselves.

Uses the langfuse v3+ SDK: `Langfuse(...)` initializes a global client that
`CallbackHandler()` picks up implicitly.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from law_ai.config import LangfuseSettings
from law_ai.logging import get_logger

logger = get_logger(__name__)


class LangfuseService:
    def __init__(self, settings: LangfuseSettings) -> None:
        self._settings = settings
        self._client: Any = None
        self._handler: Any = None

    def callback_handler(self) -> Any | None:
        """LangChain callback handler, or None when tracing is disabled."""
        if not self._settings.enabled:
            return None
        if self._handler is None:
            from langfuse import Langfuse  # lazy import
            from langfuse.langchain import CallbackHandler

            self._client = Langfuse(
                public_key=self._settings.public_key,
                secret_key=self._settings.secret_key,
                host=self._settings.host,
            )
            self._handler = CallbackHandler()
            logger.info(
                "langfuse.enabled",
                host=self._settings.host,
                auth_ok=self._client.auth_check(),
            )
        return self._handler

    @contextmanager
    def span(self, name: str, *, input: Any = None) -> Iterator[Any | None]:
        """Manual span nested into the active trace (OTel context), or a no-op.

        Lets non-LangChain code (e.g. the retrieval service) appear as its own
        node in the trace tree with structured input/output — the LangChain
        callback only captures LLM/chain steps.
        """
        if not self._settings.enabled:
            yield None
            return
        self.callback_handler()  # ensures the client is initialized
        with self._client.start_as_current_observation(
            as_type="span", name=name, input=input
        ) as span:
            yield span

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.flush()
