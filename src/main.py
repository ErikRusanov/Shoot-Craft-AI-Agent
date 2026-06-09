"""Process entrypoint: configure logging, then serve the API via uvicorn.

The FastAPI app is built by the factory in `api/app.py`; this module owns only
the things that must happen once per process before it starts. Run with
`make run` (or `python -m main`).
"""

import logging

import structlog
import uvicorn

from config import Settings, get_settings


def configure_logging(settings: Settings) -> None:
    """Wire stdlib logging into structlog.

    JSON in prod (machine-parseable), pretty console in dev. No PII/biometrics
    ever reach the logs — only session_key/face_key and metrics.
    """
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


def main() -> None:
    settings = get_settings()
    configure_logging(settings)

    structlog.get_logger().info(
        "starting", env=settings.app_env, host=settings.host, port=settings.port
    )

    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_config=None,  # logging is owned by structlog, not uvicorn's dictConfig
    )


if __name__ == "__main__":
    main()
