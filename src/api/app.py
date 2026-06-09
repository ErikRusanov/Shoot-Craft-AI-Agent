"""FastAPI factory.

Builds the DI container once per app and parks it on `app.state`. Callable with
no args (uvicorn `factory=True`) — then settings come from the environment —
or with explicit `Settings` from tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.deps import build_container
from api.routes import router
from config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    container = build_container(settings or get_settings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # Don't leave graph runs dangling past the server.
        await container.runner.aclose()

    app = FastAPI(title="photocore", lifespan=lifespan)
    app.state.container = container
    app.include_router(router)
    return app
