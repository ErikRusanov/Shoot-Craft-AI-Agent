"""Walking-skeleton e2e: API → graph → events → interrupt → resume, all on fakes.

Two go/no-go properties of the LangGraph choice are pinned here:

1. interrupt + background run + SSE compose cleanly over HTTP, and
2. a resume works in a *fresh* graph instance sharing only the checkpointer —
   i.e. the process can crash between the question and the answer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import NamedTuple

import httpx
import pytest
import uvicorn
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from api.app import create_app
from config import Settings
from graph.builder import build_graph
from graph.state import GraphState
from schemas import Event
from tests.fakes.store import InMemoryEventBus

E2E_TIMEOUT = 10

START_BODY = {
    "face_key": "face-1",
    "use_case": "avatar",
    "gender": "female",
    "age": 30,
    "budget_limit": 4,
    "idem_key": "idem-start-1",
}
INPUT_BODY = {
    "session_key": "s1",
    "slot": "style",
    "value": "casual",
    "idem_key": "idem-input-1",
}


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


# httpx.ASGITransport runs the ASGI app to completion and buffers the whole
# body, so a live SSE stream can never be observed through it. The skeleton's
# whole point is proving *live* streaming, so the e2e runs a real uvicorn on an
# ephemeral port instead.
@pytest.fixture
async def server_url() -> AsyncIterator[str]:
    app = create_app(Settings(_env_file=None))
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    async with asyncio.timeout(E2E_TIMEOUT):
        while not server.started:
            if serve_task.done():
                serve_task.result()  # surface the startup failure
            await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await serve_task


async def test_skeleton_e2e(server_url: str) -> None:
    """POST start → events stream → need_input → POST input → resume → done."""
    async with (
        asyncio.timeout(E2E_TIMEOUT),
        httpx.AsyncClient(base_url=server_url, timeout=E2E_TIMEOUT) as client,
    ):
        resp = await client.post("/v1/sessions/s1", json=START_BODY)
        assert resp.status_code == 202
        assert resp.json()["accepted"] is True

        seen: list[SseFrame] = []
        async with client.stream("GET", "/v1/sessions/s1/events") as stream:
            frames = iter_sse(stream.aiter_lines())
            async for frame in frames:
                seen.append(frame)
                if frame.event == "need_input":
                    break

            question = json.loads(seen[-1].data)
            assert question["slot"] == "style"
            assert question["schema_v"] == 1

            resp = await client.post("/v1/sessions/s1/input", json=INPUT_BODY)
            assert resp.status_code == 202

            async for frame in frames:
                seen.append(frame)
                if frame.event == "done":
                    break

        assert [f.event for f in seen] == ["stage", "need_input", "stage", "stage", "done"]
        stages = [json.loads(f.data)["stage"] for f in seen if f.event == "stage"]
        assert stages == ["face_check", "planning", "generating"]
        # The resume value travelled through the graph to the terminal event.
        assert json.loads(seen[-1].data)["detail"] == "casual"
        # Stream ids are monotonically increasing — Last-Event-ID is resumable.
        ids = [int(f.id) for f in seen]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)

        # Reconnect after the question: only the post-interrupt tail replays.
        async with client.stream(
            "GET",
            "/v1/sessions/s1/events",
            headers={"Last-Event-ID": seen[1].id},
        ) as stream:
            replayed = [f async for f in iter_sse(stream.aiter_lines())]
        assert [f.id for f in replayed] == [f.id for f in seen[2:]]


async def test_resume_survives_graph_restart() -> None:
    """Crash simulation: a new graph instance on the same checkpointer resumes
    from the interrupt instead of replaying the pipeline."""
    bus = InMemoryEventBus()
    checkpointer = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": "s-crash"}}
    initial: GraphState = {"session_key": "s-crash", "face_key": "face-1", "slots": {}}

    async with asyncio.timeout(E2E_TIMEOUT):
        first = build_graph(bus, checkpointer=checkpointer)
        paused = await first.ainvoke(initial, config=config)
        assert paused["__interrupt__"][0].value["slot"] == "style"

        # "Restart": same checkpointer, brand-new compiled graph.
        second = build_graph(bus, checkpointer=checkpointer)
        finished = await second.ainvoke(Command(resume="studio"), config=config)
        assert finished["slots"] == {"style": "studio"}
        assert "__interrupt__" not in finished

        events: list[Event] = []
        async for item in bus.tail("s-crash"):
            events.append(item.event)
            if item.event.type == "done":
                break

    # Pre-interrupt nodes did not re-run on resume: face_check narrated once.
    assert [e.type for e in events] == ["stage", "stage", "stage", "done"]
    assert events[0].type == "stage" and events[0].stage.value == "face_check"
