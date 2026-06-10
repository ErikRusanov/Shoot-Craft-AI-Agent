"""HTTP contract — the endpoints the business service drives the core through.

Mutations return 202 with a lean `SessionAck`: the actual work runs in the
background and narrates itself over the event stream. Step-9 scope: the FSM and
its persistence are real, but the API surface is still thin — idempotency
replay, session-existence checks and the ingest endpoint land with API
hardening.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette import EventSourceResponse

from api.deps import Container
from api.sse import session_event_stream
from schemas import (
    ApproveRequest,
    FsmState,
    InputAnswerRequest,
    SessionAck,
    StartSessionRequest,
)

router = APIRouter(prefix="/v1/sessions")


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


@router.post("/{session_key}", status_code=202)
async def start_session(
    session_key: str, body: StartSessionRequest, container: ContainerDep
) -> SessionAck:
    container.runner.start(
        session_key,
        face_key=body.face_key,
        use_case=body.use_case,
        gender=body.gender,
        budget_limit=body.budget_limit,
    )
    return SessionAck(session_key=session_key, fsm_state=FsmState.CREATED)


@router.post("/{session_key}/input", status_code=202)
async def submit_input(
    session_key: str, body: InputAnswerRequest, container: ContainerDep
) -> SessionAck:
    if body.session_key != session_key:
        raise HTTPException(status_code=409, detail="session_key mismatch")
    container.runner.resume(session_key, body.value)
    return SessionAck(session_key=session_key, fsm_state=FsmState.PLANNING)


@router.post("/{session_key}/approve", status_code=202)
async def approve_plan(
    session_key: str, body: ApproveRequest, container: ContainerDep
) -> SessionAck:
    if body.session_key != session_key:
        raise HTTPException(status_code=409, detail="session_key mismatch")
    container.runner.resume(
        session_key, {"approved": body.approved, "composition_id": body.composition_id}
    )
    next_state = FsmState.GENERATING if body.approved else FsmState.FAILED
    return SessionAck(session_key=session_key, fsm_state=next_state)


@router.get("/{session_key}/events")
async def stream_events(
    session_key: str, request: Request, container: ContainerDep
) -> EventSourceResponse:
    last_id = request.headers.get("Last-Event-ID")
    return EventSourceResponse(session_event_stream(container.bus, session_key, last_id=last_id))
