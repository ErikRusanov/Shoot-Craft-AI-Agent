"""Process entrypoint: configure logging, then serve the API via uvicorn.

The FastAPI app is built by the factory in `api/app.py`; this module owns only
the things that must happen once per process before it starts. Run with
`make run` (or `python -m main`).
"""

import structlog
import uvicorn

from config import get_settings
from utils.logging import configure_logging


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
