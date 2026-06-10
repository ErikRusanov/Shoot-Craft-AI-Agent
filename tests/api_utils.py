"""Shared HTTP/SSE test plumbing for the API e2e suites.

httpx.ASGITransport runs the ASGI app to completion and buffers the whole
body, so a live SSE stream can never be observed through it — tests that need
the stream run a real uvicorn on an ephemeral port via :func:`serve`.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, NamedTuple

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image

from config import Settings

STARTUP_TIMEOUT = 30


def noise_png(side: int = 1024) -> bytes:
    """A noise frame: passes the gate honestly (sharp, lit, large) on real metrics."""
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def make_settings(tmp_path: Any, redis_url: str | None, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        fake_connectors=True,
        redis_url=redis_url,
        local_storage_path=str(tmp_path / "storage"),
        **overrides,
    )


class SseFrame(NamedTuple):
    id: str
    event: str
    data: str


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[SseFrame]:
    """Minimal SSE wire parser: id/event/data fields, blank-line dispatch."""
    frame_id, event = "", ""
    data: list[str] = []
    async for line in lines:
        if not line.strip():
            if event:
                yield SseFrame(frame_id, event, "\n".join(data))
                frame_id, event, data = "", "", []
            continue
        if line.startswith(":"):  # ping comment
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "id":
            frame_id = value
        elif field == "event":
            event = value
        elif field == "data":
            data.append(value)


async def collect_until(frames: AsyncIterator[SseFrame], seen: list[SseFrame], stop: str) -> None:
    async for frame in frames:
        seen.append(frame)
        if frame.event == stop:
            return
    raise AssertionError(f"stream ended before a {stop!r} frame")


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncIterator[str]:
    """Run ``app`` on a real uvicorn at an ephemeral port; yield the base URL."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(STARTUP_TIMEOUT):
            while not server.started:
                if serve_task.done():
                    serve_task.result()  # surface the startup failure
                await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await serve_task
