"""Logging setup — structlog over stdlib, one call per process.

No PII/biometrics ever reach the logs — bind only ``session_key``/``face_key``
and metrics. JSON renderer in prod (machine-parseable), pretty console in dev.
"""

from __future__ import annotations

import logging

import structlog

from config import Settings


def configure_logging(settings: Settings) -> None:
    """Wire stdlib logging into structlog."""
    logging.basicConfig(format="%(message)s", level=settings.log_level)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
