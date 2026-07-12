"""Structured logging via structlog.

Call `setup_logging()` once from the lifespan. In local dev logs are pretty
console lines; elsewhere they are JSON (machine-parseable, ship-anywhere).
"""

import logging
import sys

import structlog

from law_ai.config import Settings


def setup_logging(settings: Settings) -> None:
    level = getattr(logging, settings.app.log_level.upper(), logging.INFO)

    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,  # request_id from middleware
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    renderer: structlog.types.Processor
    if settings.app.env == "local":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
