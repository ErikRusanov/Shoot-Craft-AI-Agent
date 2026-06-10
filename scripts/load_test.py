"""Load driver: N full photoshoot sessions in parallel over the public API.

Drives the real contract end to end — ingest → start → SSE → input → approve →
terminal — against a running core (point it at `make run` / `make up` with
``FAKE_CONNECTORS=true`` for a free run). Reports outcome counts and the
latency distribution of the *whole session*, which is what the backpressure
knobs (semaphore, wall clock) actually shape.

    PYTHONPATH=src uv run python scripts/load_test.py --sessions 50 --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image

SESSION_TIMEOUT = 120.0


def noise_png(side: int = 1024) -> bytes:
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class Outcome:
    key: str
    status: str  # done | failed | error
    seconds: float
    detail: str = ""


async def _sse_events(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, dict[str, object]]]:
    """Yield (event, payload) pairs from an SSE byte stream."""
    event = ""
    data: list[str] = []
    async for line in lines:
        if not line.strip():
            if event:
                yield event, json.loads("\n".join(data))
                event, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)


async def drive_session(
    client: httpx.AsyncClient, run_id: str, n: int, photo_b64: str, budget: int
) -> Outcome:
    face_key = f"load-{run_id}-face-{n}"
    session_key = f"load-{run_id}-sess-{n}"
    started = time.perf_counter()

    def done(status: str, detail: str = "") -> Outcome:
        return Outcome(session_key, status, time.perf_counter() - started, detail)

    try:
        async with asyncio.timeout(SESSION_TIMEOUT):
            resp = await client.post(
                f"/v1/faces/{face_key}",
                json={"image_b64": photo_b64, "idem_key": f"{face_key}-ingest"},
            )
            resp.raise_for_status()
            if not resp.json()["accepted"]:
                return done("error", f"photo rejected: {resp.json()['gate_reason']}")

            resp = await client.post(
                f"/v1/sessions/{session_key}",
                json={
                    "face_key": face_key,
                    "use_case": "avatar",
                    "gender": "female",
                    "budget_limit": budget,
                    "idem_key": f"{session_key}-start",
                },
            )
            resp.raise_for_status()

            answer_n = 0
            async with client.stream("GET", f"/v1/sessions/{session_key}/events") as stream:
                async for event, payload in _sse_events(stream.aiter_lines()):
                    if event == "need_input" and payload["slot"] == "approve":
                        resp = await client.post(
                            f"/v1/sessions/{session_key}/approve",
                            json={
                                "session_key": session_key,
                                "approved": True,
                                "idem_key": f"{session_key}-approve",
                            },
                        )
                        resp.raise_for_status()
                    elif event == "need_input":
                        answer_n += 1
                        raw_options = payload.get("options")
                        options = raw_options if isinstance(raw_options, list) else []
                        value = str(options[0]) if options else "a plain studio scene"
                        resp = await client.post(
                            f"/v1/sessions/{session_key}/input",
                            json={
                                "session_key": session_key,
                                "slot": payload["slot"],
                                "value": value,
                                "idem_key": f"{session_key}-input-{answer_n}",
                            },
                        )
                        resp.raise_for_status()
                    elif event == "done":
                        return done("done")
                    elif event == "failed":
                        return done("failed", str(payload.get("reason", "")))
            return done("error", "stream ended without a terminal event")
    except (TimeoutError, httpx.HTTPError) as exc:
        return done("error", f"{type(exc).__name__}: {exc}")


async def run(base_url: str, sessions: int, concurrency: int, budget: int) -> int:
    run_id = uuid.uuid4().hex[:8]
    photo_b64 = base64.b64encode(noise_png()).decode()
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=SESSION_TIMEOUT) as client:
        ready = await client.get("/readyz")
        if ready.status_code != 200:
            print(f"target is not ready: {ready.status_code} {ready.text}", file=sys.stderr)
            return 2

        async def bounded(n: int) -> Outcome:
            async with semaphore:
                return await drive_session(client, run_id, n, photo_b64, budget)

        wall_start = time.perf_counter()
        outcomes = await asyncio.gather(*(bounded(n) for n in range(sessions)))
        wall = time.perf_counter() - wall_start

    by_status: dict[str, int] = {}
    for o in outcomes:
        by_status[o.status] = by_status.get(o.status, 0) + 1
    durations = sorted(o.seconds for o in outcomes)
    print(f"\nsessions={sessions} concurrency={concurrency} wall={wall:.2f}s")
    print(f"outcomes: {by_status}")
    print(
        f"latency: p50={statistics.median(durations):.3f}s "
        f"p95={durations[max(0, int(len(durations) * 0.95) - 1)]:.3f}s "
        f"max={durations[-1]:.3f}s"
    )
    print(f"throughput: {sessions / wall:.2f} sessions/s")
    for o in outcomes:
        if o.status != "done":
            print(f"  {o.key}: {o.status} {o.detail}", file=sys.stderr)
    return 0 if by_status.get("done", 0) == sessions else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    return asyncio.run(run(args.base_url, args.sessions, args.concurrency, args.budget))


if __name__ == "__main__":
    sys.exit(main())
