"""FastAPI factory.

Builds the DI container once per app and parks it on `app.state`. Callable with
no args (uvicorn `factory=True`) — then settings come from the environment —
or with explicit `Settings` from tests.

Owns the Redis-loss policy at the HTTP edge: a store/bus connection failure
inside a request becomes a 503, never a silent failover to in-memory state
(replicas would split-brain). `/readyz` reports the same condition to the
orchestrator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from api.deps import build_container
from api.routes import faces_router, health_router, router
from config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    container = build_container(settings or get_settings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Checkpointer indices and similar async-only init happen here — the
        # factory itself stays sync for uvicorn's `factory=True`.
        await container.astart()
        yield
        # Don't leave graph runs or connector clients dangling past the server.
        await container.aclose()

    app = FastAPI(title="photocore", lifespan=lifespan)
    app.state.container = container
    app.include_router(router)
    app.include_router(faces_router)
    app.include_router(health_router)

    async def _store_unavailable(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": "state store unavailable"}, status_code=503)

    app.add_exception_handler(RedisConnectionError, _store_unavailable)
    app.add_exception_handler(RedisTimeoutError, _store_unavailable)
    return app
