"""Logging setup — structlog over stdlib, one call per process.

No PII/biometrics ever reach the logs — bind only ``session_key``/``face_key``
and metrics. Discipline alone is not a guarantee, so a scrub processor sits in
the chain as the backstop: any bound value under a sensitive key name is
redacted before rendering, recursively. JSON renderer in prod
(machine-parseable), pretty console in dev.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

import structlog

from config import Settings

# Key names whose values must never render: biometrics (embedding), raw image
# payloads, user-typed text (answer/value — the free-form scene is user
# content), prompts (carry the scene text), and credentials. Matching is by
# exact lowercased key — a metrics key like `n_images` stays loggable.
_SENSITIVE_KEYS = frozenset(
    {
        "embedding",
        "image",
        "image_b64",
        "photo",
        "face_crop",
        "reference_images",
        "answer",
        "value",
        "prompt",
        "api_key",
        "authorization",
        "secret",
    }
)
_REDACTED = "[redacted]"


def _scrub[M: MutableMapping[str, Any]](mapping: M) -> M:
    for key, value in mapping.items():
        if key.lower() in _SENSITIVE_KEYS:
            mapping[key] = _REDACTED
        elif isinstance(value, dict):
            _scrub(value)
    return mapping


def scrub_sensitive(
    logger: structlog.types.WrappedLogger, method_name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Structlog processor: redact sensitive keys in the event (recursively)."""
    return _scrub(event_dict)


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
            scrub_sensitive,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
