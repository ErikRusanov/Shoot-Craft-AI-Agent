"""Losing Redis at runtime is an outage, not a failover.

The wiring decision (Redis vs in-memory) is made once, at process start. When
Redis dies under a running app the contract is: ``/readyz`` flips to 503 so
the orchestrator stops routing, every store-touching request answers 503, and
the process **never** degrades to in-memory state (replicas would split-brain).

This test owns its Redis container (the session-scoped one in conftest is
shared with other tests and must stay alive), kills it mid-flight and probes
the surface. Skipped without Docker, like every container-backed test.
"""

from __future__ import annotations

import asyncio
import base64

import httpx
import pytest

from api.app import create_app
from tests.api_utils import make_settings, noise_png, serve
from tests.conftest import REDIS_IMAGE

TIMEOUT = 60


async def test_redis_death_is_503_not_failover(tmp_path: object) -> None:
    testcontainers = pytest.importorskip("testcontainers.redis")
    redis_container = testcontainers.RedisContainer(REDIS_IMAGE)
    try:
        redis_container.start()
    except Exception:
        pytest.skip("Docker is not available — skipping Redis-backed tests")

    stopped = False

    def stop_redis() -> None:
        nonlocal stopped
        if not stopped:
            stopped = True
            redis_container.stop()

    try:
        host = redis_container.get_container_host_ip()
        port = redis_container.get_exposed_port(redis_container.port)
        settings = make_settings(tmp_path, f"redis://{host}:{port}/0")
        app = create_app(settings)
        async with (
            asyncio.timeout(TIMEOUT),
            serve(app) as url,
            httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as client,
        ):
            # Healthy baseline: ready, and a real mutation lands in Redis.
            resp = await client.get("/readyz")
            assert resp.status_code == 200
            assert resp.json()["redis"] == {"mode": "redis", "ok": True}
            resp = await client.post(
                "/v1/faces/f1",
                json={"image_b64": base64.b64encode(noise_png()).decode(), "idem_key": "i1"},
            )
            assert resp.status_code == 200

            stop_redis()

            # Liveness stays green (the process is fine); readiness goes red.
            assert (await client.get("/healthz")).status_code == 200
            resp = await client.get("/readyz")
            assert resp.status_code == 503
            assert resp.json()["redis"] == {"mode": "redis", "ok": False}

            # Mutations and store-backed reads refuse instead of degrading.
            resp = await client.post(
                "/v1/faces/f2",
                json={"image_b64": base64.b64encode(noise_png()).decode(), "idem_key": "i2"},
            )
            assert resp.status_code == 503
            assert (await client.get("/v1/sessions/s1")).status_code == 503
    finally:
        stop_redis()
