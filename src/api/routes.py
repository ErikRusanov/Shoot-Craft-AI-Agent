"""HTTP contract — the endpoints the business service drives the core through.

Mutations return 202 with a lean ack: the actual work runs in the background
and narrates itself over the event stream. Hardening rules, uniform across the
surface:

- **Idempotency.** Every mutating request carries ``idem_key``; the response
  of the first execution is stored (step-4 store, ``http:`` namespace) and a
  retried call replays it byte-for-byte. Only *committed* outcomes are stored —
  a rejection (404/409/4xx) is never recorded, so a retry after the blocker
  clears still executes.
- **Conflicts are statuses, not acks.** A mutation that cannot apply (wrong
  FSM stage, a run already in flight, a duplicate start) is a 409 with a
  reason, a missing aggregate is a 404. ``SessionAck`` is reserved for
  accepted work.
- **One run at a time.** start/input/approve spawn through the runner's
  distributed run lock; the 409 surfaces concurrent drives of one session.
- **Redis loss is a 503.** The app-level handler (api/app.py) maps store
  connection failures to 503 — no silent in-memory failover, ever.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from api.deps import Container
from api.sse import session_event_stream
from schemas import (
    ApproveRequest,
    FsmState,
    IngestPhotoRequest,
    IngestPhotoResponse,
    InputAnswerRequest,
    Preset,
    PresetAsk,
    PresetCatalog,
    PresetSummary,
    SchemaModel,
    SessionAck,
    SessionSnapshot,
    StartSessionRequest,
    StartSessionResponse,
    Verdict,
)
from services.classifier import FALLBACK_USE_CASE
from services.vision import face_crop_ref, photo_ref

router = APIRouter(prefix="/v1/sessions")
faces_router = APIRouter(prefix="/v1/faces")
presets_router = APIRouter(prefix="/v1/presets")
health_router = APIRouter()

_TERMINAL_STATES = frozenset({FsmState.DONE, FsmState.FAILED, FsmState.CANCELLED})


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


async def _idempotent[M: SchemaModel](
    container: Container,
    scope: str,
    idem_key: str,
    model: type[M],
    op: Callable[[], Awaitable[M]],
) -> M:
    """Run ``op`` once per ``idem_key``, replay its response ever after.

    The namespace keeps HTTP records apart from internal ones (the generation
    loop owns ``generation:*``); the scope keeps one business ``idem_key`` from
    colliding across endpoints. ``op`` signals a non-committed outcome by
    raising (HTTPException) — exceptions are never stored, only responses.
    """

    async def _serialized() -> bytes:
        return (await op()).model_dump_json().encode()

    payload, _replayed = await container.idempotency.run_once(
        f"http:{scope}:{idem_key}",
        ttl_seconds=container.settings.session_ttl_seconds,
        op=_serialized,
    )
    return model.model_validate_json(payload)


async def _session_or_404(container: Container, session_key: str) -> Any:
    session = await container.store.get_session(session_key)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_key!r}")
    return session


# --- faces ---


@faces_router.post("/{face_key}")
async def ingest_photo(
    face_key: str, body: IngestPhotoRequest, container: ContainerDep
) -> IngestPhotoResponse:
    """Ingest one reference photo: store it, gate it, build the face profile."""
    try:
        image = base64.b64decode(body.image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64") from exc
    if len(image) > container.settings.max_photo_bytes:
        raise HTTPException(status_code=413, detail="photo exceeds the size limit")
    if not image:
        raise HTTPException(status_code=400, detail="image_b64 decodes to an empty payload")

    async def op() -> IngestPhotoResponse:
        try:
            async with asyncio.timeout(container.settings.ingest_timeout_seconds):
                await container.storage.put(photo_ref(face_key), image)
                ingest = await container.vision.ingest(
                    image, face_key=face_key, photo_ref=photo_ref(face_key)
                )
                await container.store.put_face(
                    ingest.profile, ttl_seconds=container.settings.face_ttl_seconds
                )
                if ingest.face_crop is not None:
                    await container.storage.put(face_crop_ref(face_key), ingest.face_crop)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="photo analysis timed out") from exc
        except ValueError as exc:
            # utils.images.decode_rgb's one error type for "not an image".
            raise HTTPException(status_code=400, detail="payload is not a decodable image") from exc
        profile = ingest.profile
        return IngestPhotoResponse(
            face_key=face_key,
            # Below-floor or no identity vector → the photo cannot anchor a
            # session; the caller asks the user to re-shoot.
            accepted=profile.gate_verdict is not Verdict.BELOW_FLOOR and bool(profile.embedding),
            gate_verdict=profile.gate_verdict,
            gate_reason=profile.gate_reason,
            metrics=profile.metrics,
        )

    return await _idempotent(container, "ingest", body.idem_key, IngestPhotoResponse, op)


# --- sessions ---


@router.post("/{session_key}", status_code=202)
async def start_session(
    session_key: str, body: StartSessionRequest, container: ContainerDep
) -> StartSessionResponse:
    async def op() -> StartSessionResponse:
        if await container.store.get_session(session_key) is not None:
            raise HTTPException(status_code=409, detail=f"session {session_key!r} already exists")
        face = await container.store.get_face(body.face_key)
        if face is None:
            # No profile yet — fine if the raw photo is in storage (the graph
            # ingests lazily); nothing at all is a caller error worth a 404.
            try:
                await container.storage.get(photo_ref(body.face_key))
            except KeyError as exc:
                raise HTTPException(
                    status_code=404, detail=f"unknown face_key {body.face_key!r}; ingest first"
                ) from exc
        elif face.gate_verdict is Verdict.BELOW_FLOOR or not face.embedding:
            # The profile exists but the photo cannot anchor an identity: reject
            # up front (symmetric to ingest's `accepted`), don't spawn a run that
            # would only fail in the gate node.
            raise HTTPException(
                status_code=422,
                detail=f"face_key {body.face_key!r} did not pass the quality gate; re-ingest",
            )
        # With a use_case the preset (and its id) are known up front; without one
        # the classify node picks it from the brief, so the response can only
        # confirm a run will start (a fallback exists), not which preset yet.
        preset = (
            container.library.resolve(use_case=body.use_case)
            if body.use_case
            else container.library.fallback
        )
        started = await container.runner.start(
            session_key,
            face_key=body.face_key,
            use_case=body.use_case or "",
            brief=body.brief or "",
            budget_limit=body.budget_limit,
        )
        if not started:
            raise HTTPException(status_code=409, detail="a run for this session is in flight")
        # No preset and no fallback: the spawned run fails the session on the
        # stream; the response says so up front.
        return StartSessionResponse(
            session_key=session_key,
            fsm_state=FsmState.CREATED if preset is not None else FsmState.FAILED,
            matched=preset is not None,
            preset_id=preset.id if (body.use_case and preset is not None) else None,
        )

    return await _idempotent(container, "start", body.idem_key, StartSessionResponse, op)


@router.post("/{session_key}/input", status_code=202)
async def submit_input(
    session_key: str, body: InputAnswerRequest, container: ContainerDep
) -> SessionAck:
    if body.session_key != session_key:
        raise HTTPException(status_code=409, detail="session_key mismatch")

    async def op() -> SessionAck:
        session = await _session_or_404(container, session_key)
        if session.fsm_state is not FsmState.NEED_INPUT:
            raise HTTPException(
                status_code=409,
                detail=f"session is not awaiting input (state: {session.fsm_state})",
            )
        if not await container.runner.resume(session_key, body.value):
            raise HTTPException(status_code=409, detail="a run for this session is in flight")
        return SessionAck(session_key=session_key, fsm_state=FsmState.PLANNING)

    return await _idempotent(container, "input", body.idem_key, SessionAck, op)


@router.post("/{session_key}/approve", status_code=202)
async def approve_plan(
    session_key: str, body: ApproveRequest, container: ContainerDep
) -> SessionAck:
    if body.session_key != session_key:
        raise HTTPException(status_code=409, detail="session_key mismatch")

    async def op() -> SessionAck:
        session = await _session_or_404(container, session_key)
        if session.fsm_state is not FsmState.AWAITING_APPROVAL:
            raise HTTPException(
                status_code=409,
                detail=f"session is not awaiting approval (state: {session.fsm_state})",
            )
        resumed = await container.runner.resume(
            session_key, {"approved": body.approved, "composition_id": body.composition_id}
        )
        if not resumed:
            raise HTTPException(status_code=409, detail="a run for this session is in flight")
        next_state = FsmState.GENERATING if body.approved else FsmState.FAILED
        return SessionAck(session_key=session_key, fsm_state=next_state)

    return await _idempotent(container, "approve", body.idem_key, SessionAck, op)


@router.post("/{session_key}/cancel", status_code=202)
async def cancel_session(session_key: str, container: ContainerDep) -> SessionAck:
    """Caller-initiated terminal. Naturally idempotent — no ``idem_key``:
    cancelling twice finds a terminal session and reports 409."""
    session = await _session_or_404(container, session_key)
    if session.fsm_state in _TERMINAL_STATES:
        raise HTTPException(
            status_code=409, detail=f"session is already terminal (state: {session.fsm_state})"
        )
    await container.runner.cancel(session_key)
    return SessionAck(session_key=session_key, fsm_state=FsmState.CANCELLED)


@router.get("/{session_key}")
async def session_snapshot(session_key: str, container: ContainerDep) -> SessionSnapshot:
    session = await _session_or_404(container, session_key)
    return SessionSnapshot(
        state=session,
        generations_spent=session.generations_spent(),
        cost_spent=session.cost_spent(),
    )


@router.get("/{session_key}/events")
async def stream_events(
    session_key: str, request: Request, container: ContainerDep
) -> EventSourceResponse:
    last_id = request.headers.get("Last-Event-ID")
    return EventSourceResponse(session_event_stream(container.bus, session_key, last_id=last_id))


# --- presets ---


def _preset_summary(preset: Preset) -> PresetSummary:
    """The catalog projection of a preset — matcher tokens and its ask slot only,
    never the frozen prompt content (the moat)."""
    asks = [
        PresetAsk(
            slot=name,
            options=[str(o) for o in slot.enum] if slot.enum else None,
            default=str(slot.default) if slot.default is not None else None,
        )
        for name, slot in preset.slots.items()
        if slot.ask
    ]
    return PresetSummary(
        id=preset.id,
        version=preset.version,
        use_case_tokens=list(preset.applies_to.use_case),
        is_fallback=preset.id == FALLBACK_USE_CASE,
        asks=asks,
    )


@presets_router.get("")
async def list_presets(container: ContainerDep) -> PresetCatalog:
    """The preset catalog the business service offers the user — ids, versions,
    matcher tokens and ask slots. The reserved ``default`` fallback is flagged,
    not offered as a choice."""
    presets = [
        _preset_summary(preset)
        for pid in container.library.ids
        if (preset := container.library.get(pid)) is not None
    ]
    return PresetCatalog(presets=presets, library_version=container.library.library_version)


# --- health ---


@health_router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process serves requests. No dependencies touched."""
    return {"status": "ok"}


@health_router.get("/readyz")
async def readyz(container: ContainerDep) -> JSONResponse:
    """Readiness: can this replica do real work *right now*?

    Reflects the two startup-critical dependencies: the preset library (loaded
    once, immutable) and Redis when configured. Redis loss flips this to 503 —
    the orchestrator should stop routing here; there is no in-memory failover.
    """
    redis_ok = True
    if container.redis is None:
        redis_check: dict[str, Any] = {"mode": "in-memory"}
    else:
        try:
            # redis-py types ping() as a sync/async union; the asyncio client
            # always returns an awaitable.
            await cast("Awaitable[bool]", container.redis.ping())
            redis_check = {"mode": "redis", "ok": True}
        except Exception:
            redis_ok = False
            redis_check = {"mode": "redis", "ok": False}
    body = {
        "status": "ready" if redis_ok else "unavailable",
        "redis": redis_check,
        "presets": {
            "count": len(container.library),
            "library_version": container.library.library_version,
            "source": container.library.source,
        },
    }
    return JSONResponse(body, status_code=200 if redis_ok else 503)
