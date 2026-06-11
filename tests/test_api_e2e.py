"""API hardening e2e — the contract surface over HTTP, fake connectors, in-memory.

What this suite pins (the happy-path lifecycle itself lives in
tests/test_graph_e2e.py and runs on both wirings):

- the ingest endpoint: gate outcome on the response, profile stored, byte-true
  idempotent replay (a retry with the same ``idem_key`` never redoes the work);
- start: replay vs duplicate (replayed body vs 409), unknown ``face_key`` 404;
- stage guards: input/approve/cancel against the wrong FSM stage are 409s,
  unknown sessions 404s;
- cancel: interrupts a waiting session, closes the stream with a terminal
  event, and blocks any further drive of the FSM;
- the run lock: while a run is in flight no second mutation can drive the
  same session;
- the wall-clock ceiling: a run that exceeds it fails the session cleanly;
- health/readiness on the in-memory wiring.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from api.app import create_app
from api.deps import Container
from schemas import FsmState
from tests.api_utils import SseFrame, collect_until, iter_sse, make_settings, noise_png, serve

TIMEOUT = 30

GOOD_PHOTO_B64 = base64.b64encode(noise_png()).decode()
SMALL_PHOTO_B64 = base64.b64encode(noise_png(side=200)).decode()

START_BODY = {
    "face_key": "f1",
    "use_case": "avatar",
    "budget_limit": 4,
    "idem_key": "start-1",
}


@pytest.fixture
async def server(tmp_path: Any) -> AsyncIterator[tuple[str, Container]]:
    app = create_app(make_settings(tmp_path, None))
    async with serve(app) as url:
        yield url, app.state.container


async def _ingest(client: httpx.AsyncClient, face_key: str = "f1", **kw: Any) -> httpx.Response:
    body = {"image_b64": GOOD_PHOTO_B64, "idem_key": f"ingest-{face_key}"} | kw
    return await client.post(f"/v1/faces/{face_key}", json=body)


async def _drive_to_need_input(
    client: httpx.AsyncClient, session_key: str, body: dict[str, Any]
) -> None:
    """Start a session and block until it parks on the ask interrupt."""
    resp = await client.post(f"/v1/sessions/{session_key}", json=body)
    assert resp.status_code == 202, resp.text
    seen: list[SseFrame] = []
    async with client.stream("GET", f"/v1/sessions/{session_key}/events") as stream:
        await collect_until(iter_sse(stream.aiter_lines()), seen, "need_input")


async def test_ingest_gate_and_idempotent_replay(server: tuple[str, Container]) -> None:
    url, container = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        resp = await _ingest(c)
        assert resp.status_code == 200
        first = resp.json()
        assert first["accepted"] is True
        assert first["gate_verdict"] == "passed"
        assert first["gate_reason"] == "ok"
        assert first["metrics"]["width"] == 1024
        assert await container.store.get_face("f1") is not None

        # Same idem_key, different (failing) payload: the stored response
        # replays byte-true — the gate does not run again.
        resp = await _ingest(c, image_b64=SMALL_PHOTO_B64)
        assert resp.status_code == 200
        assert resp.json() == first

        # A fresh idem_key with the failing payload really is re-evaluated.
        resp = await _ingest(c, "f-low", image_b64=SMALL_PHOTO_B64, idem_key="ingest-low")
        low = resp.json()
        assert low["accepted"] is False
        assert low["gate_verdict"] == "below_floor"
        assert low["gate_reason"] == "low_resolution"


async def test_ingest_rejects_bad_payloads(server: tuple[str, Container]) -> None:
    url, _ = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        resp = await c.post(
            "/v1/faces/f-bad", json={"image_b64": "not-base64!!!", "idem_key": "bad-1"}
        )
        assert resp.status_code == 400
        resp = await c.post("/v1/faces/f-bad", json={"image_b64": "", "idem_key": "bad-2"})
        assert resp.status_code == 400
        not_an_image = base64.b64encode(b"plain text, definitely not pixels").decode()
        resp = await c.post(
            "/v1/faces/f-bad", json={"image_b64": not_an_image, "idem_key": "bad-3"}
        )
        assert resp.status_code == 400


async def test_ingest_size_cap(tmp_path: Any) -> None:
    app = create_app(make_settings(tmp_path, None, max_photo_bytes=1000))
    async with (
        asyncio.timeout(TIMEOUT),
        serve(app) as url,
        httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c,
    ):
        resp = await _ingest(c)
        assert resp.status_code == 413


async def test_full_flow_snapshot_and_terminal_guards(server: tuple[str, Container]) -> None:
    """ingest → start → input → approve → done → snapshot; then the terminal
    session rejects further mutations."""
    url, _ = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        assert (await _ingest(c)).json()["accepted"] is True

        resp = await c.post("/v1/sessions/s1", json=START_BODY)
        assert resp.status_code == 202
        started = resp.json()
        assert started["matched"] is True
        assert started["preset_id"] == "demo_avatar"
        assert started["fsm_state"] == "created"

        seen: list[SseFrame] = []
        async with c.stream("GET", "/v1/sessions/s1/events") as stream:
            frames = iter_sse(stream.aiter_lines())
            await collect_until(frames, seen, "need_input")
            resp = await c.post(
                "/v1/sessions/s1/input",
                json={
                    "session_key": "s1",
                    "slot": "purpose",
                    "value": "a chat or forum avatar",
                    "idem_key": "input-1",
                },
            )
            assert resp.status_code == 202
            await collect_until(frames, seen, "need_input")
            resp = await c.post(
                "/v1/sessions/s1/approve",
                json={"session_key": "s1", "approved": True, "idem_key": "approve-1"},
            )
            assert resp.status_code == 202
            await collect_until(frames, seen, "done")

        resp = await c.get("/v1/sessions/s1")
        assert resp.status_code == 200
        snapshot = resp.json()
        assert snapshot["state"]["fsm_state"] == "done"
        assert snapshot["state"]["preset_id"] == "demo_avatar"
        assert snapshot["generations_spent"] == 1

        # Terminal: no further drive, in any direction.
        resp = await c.post("/v1/sessions/s1/cancel")
        assert resp.status_code == 409
        resp = await c.post(
            "/v1/sessions/s1/input",
            json={"session_key": "s1", "slot": "purpose", "value": "x", "idem_key": "input-2"},
        )
        assert resp.status_code == 409


async def test_start_replay_vs_duplicate(server: tuple[str, Container]) -> None:
    url, _ = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        await _ingest(c)
        await _drive_to_need_input(c, "s1", START_BODY)

        # Same idem_key → the stored response replays; the FSM is not re-driven.
        first = await c.post("/v1/sessions/s1", json=START_BODY)
        assert first.status_code == 202
        assert first.json()["fsm_state"] == "created"

        # New idem_key for an existing session → a real duplicate → conflict.
        resp = await c.post("/v1/sessions/s1", json=START_BODY | {"idem_key": "start-2"})
        assert resp.status_code == 409

        # The replay did not restart the run: exactly one face_check stage.
        seen: list[SseFrame] = []
        async with c.stream("GET", "/v1/sessions/s1/events") as stream:
            await collect_until(iter_sse(stream.aiter_lines()), seen, "need_input")
        stages = [json.loads(f.data)["stage"] for f in seen if f.event == "stage"]
        assert stages.count("face_check") == 1


async def test_start_on_below_floor_face_is_422(server: tuple[str, Container]) -> None:
    url, _ = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        # Ingest a photo the gate rejects as below-floor (low resolution).
        low = await _ingest(c, "f-low", image_b64=SMALL_PHOTO_B64, idem_key="ingest-low")
        assert low.json()["gate_verdict"] == "below_floor"

        # Starting on it is rejected up front (fail-fast), not spawned to fail
        # later in the gate node — symmetric to ingest's `accepted: false`.
        resp = await c.post(
            "/v1/sessions/slow",
            json=START_BODY | {"face_key": "f-low", "idem_key": "start-low"},
        )
        assert resp.status_code == 422, resp.text


async def test_unknown_aggregates_are_404(server: tuple[str, Container]) -> None:
    url, _ = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        resp = await c.post("/v1/sessions/ghost", json=START_BODY | {"face_key": "ghost-face"})
        assert resp.status_code == 404

        assert (await c.get("/v1/sessions/ghost")).status_code == 404
        assert (await c.post("/v1/sessions/ghost/cancel")).status_code == 404
        resp = await c.post(
            "/v1/sessions/ghost/input",
            json={"session_key": "ghost", "slot": "s", "value": "v", "idem_key": "i"},
        )
        assert resp.status_code == 404
        resp = await c.post(
            "/v1/sessions/ghost/approve",
            json={"session_key": "ghost", "approved": True, "idem_key": "i"},
        )
        assert resp.status_code == 404


async def test_wrong_stage_is_409(server: tuple[str, Container]) -> None:
    url, _ = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        await _ingest(c)
        await _drive_to_need_input(c, "s1", START_BODY)
        # The session awaits the ask slot, not approval.
        resp = await c.post(
            "/v1/sessions/s1/approve",
            json={"session_key": "s1", "approved": True, "idem_key": "approve-early"},
        )
        assert resp.status_code == 409
        # And a mismatched path/body pair never reaches the FSM.
        resp = await c.post(
            "/v1/sessions/s1/input",
            json={"session_key": "other", "slot": "purpose", "value": "x", "idem_key": "i"},
        )
        assert resp.status_code == 409


async def test_cancel_interrupts_and_blocks_resume(server: tuple[str, Container]) -> None:
    url, container = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        await _ingest(c)
        await _drive_to_need_input(c, "s1", START_BODY)

        resp = await c.post("/v1/sessions/s1/cancel")
        assert resp.status_code == 202
        assert resp.json()["fsm_state"] == "cancelled"

        # The stream closed with the terminal event.
        seen: list[SseFrame] = []
        async with c.stream("GET", "/v1/sessions/s1/events") as stream:
            await collect_until(iter_sse(stream.aiter_lines()), seen, "failed")
        assert json.loads(seen[-1].data)["reason"] == "cancelled by the caller"

        snapshot = (await c.get("/v1/sessions/s1")).json()
        assert snapshot["state"]["fsm_state"] == "cancelled"

        # Cancelled is terminal: the parked interrupt can never be resumed.
        resp = await c.post(
            "/v1/sessions/s1/input",
            json={"session_key": "s1", "slot": "purpose", "value": "x", "idem_key": "i2"},
        )
        assert resp.status_code == 409
        assert (await c.post("/v1/sessions/s1/cancel")).status_code == 409

    session = await container.store.get_session("s1")
    assert session is not None
    assert session.fsm_state is FsmState.CANCELLED


async def test_run_lock_blocks_concurrent_drive(server: tuple[str, Container]) -> None:
    url, container = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        await _ingest(c)
        await _drive_to_need_input(c, "s1", START_BODY)

        # Another replica holds the run lock (e.g. a run is mid-flight there).
        assert await container.store.acquire_lock("run:s1", token="intruder", ttl_seconds=60)
        body = {
            "session_key": "s1",
            "slot": "purpose",
            "value": "a chat or forum avatar",
            "idem_key": "input-locked",
        }
        resp = await c.post("/v1/sessions/s1/input", json=body)
        assert resp.status_code == 409
        assert "in flight" in resp.json()["detail"]

        # The blocked attempt was not recorded: after the lock clears, the
        # same idem_key executes for real.
        assert await container.store.release_lock("run:s1", token="intruder")
        resp = await c.post("/v1/sessions/s1/input", json=body)
        assert resp.status_code == 202


async def test_wall_clock_fails_session(tmp_path: Any) -> None:
    """A run past the wall-clock ceiling dies and the stream says why."""
    app = create_app(make_settings(tmp_path, None, session_wall_clock_seconds=0))
    container: Container = app.state.container
    async with (
        asyncio.timeout(TIMEOUT),
        serve(app) as url,
        httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c,
    ):
        await container.storage.put("photos/f1", noise_png())
        resp = await c.post("/v1/sessions/s1", json=START_BODY)
        assert resp.status_code == 202
        seen: list[SseFrame] = []
        async with c.stream("GET", "/v1/sessions/s1/events") as stream:
            await collect_until(iter_sse(stream.aiter_lines()), seen, "failed")
        assert json.loads(seen[-1].data)["reason"] == "session run exceeded the wall-clock limit"


async def test_list_presets(server: tuple[str, Container]) -> None:
    """The catalog advertises ids, versions, matcher tokens and ask slots; the
    reserved fallback is flagged, not offered as a choice."""
    url, _ = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        resp = await c.get("/v1/presets")
        assert resp.status_code == 200
        catalog = resp.json()
        assert catalog["library_version"] == "examples"

        by_id = {p["id"]: p for p in catalog["presets"]}
        assert "avatar" in by_id["demo_avatar"]["use_case_tokens"]
        assert by_id["demo_avatar"]["is_fallback"] is False
        # The fallback is present but flagged, and asks a free-form scene (no options).
        fallback = by_id["default"]
        assert fallback["is_fallback"] is True
        assert fallback["use_case_tokens"] == ["default"]
        assert fallback["asks"] == [{"slot": "scene", "options": None, "default": None}]


async def test_health_and_readiness(server: tuple[str, Container]) -> None:
    url, _ = server
    async with asyncio.timeout(TIMEOUT), httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        resp = await c.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

        resp = await c.get("/readyz")
        assert resp.status_code == 200
        ready = resp.json()
        assert ready["status"] == "ready"
        assert ready["redis"] == {"mode": "in-memory"}
        assert ready["presets"]["count"] == 3
        assert ready["presets"]["library_version"] == "examples"
