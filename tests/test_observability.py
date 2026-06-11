"""Log scrubbing and telemetry — the no-PII guarantees, pinned.

The audit is two-sided: the scrub processor redacts sensitive keys that would
otherwise leak into any log line (the backstop), and the telemetry event is
de-identified *by construction* — built only from preset/metric fields, never
from keys, embeddings or images.
"""

from __future__ import annotations

from decimal import Decimal

from structlog.testing import capture_logs

from schemas import (
    BestResult,
    FrameMetrics,
    FsmState,
    Iteration,
    RiskLevel,
    SessionState,
    Verdict,
)
from services.telemetry import Telemetry
from utils.logging import scrub_sensitive


def test_scrub_redacts_sensitive_keys_recursively() -> None:
    event = {
        "event": "ingest",
        "session_key": "s1",
        "face_key": "f1",
        "embedding": [0.1, 0.2],
        "image_b64": "abcd",
        "nested": {"prompt": "secret text", "blur_var": 42.0, "answer": "user text"},
        "value": "user input",
        "n_iterations": 3,
    }
    scrubbed = scrub_sensitive(None, "info", event)

    assert scrubbed["embedding"] == "[redacted]"
    assert scrubbed["image_b64"] == "[redacted]"
    assert scrubbed["value"] == "[redacted]"
    assert scrubbed["nested"]["prompt"] == "[redacted]"
    assert scrubbed["nested"]["answer"] == "[redacted]"
    # Keys and metrics stay — they are the sanctioned log vocabulary.
    assert scrubbed["session_key"] == "s1"
    assert scrubbed["face_key"] == "f1"
    assert scrubbed["nested"]["blur_var"] == 42.0
    assert scrubbed["n_iterations"] == 3


def _metrics() -> FrameMetrics:
    return FrameMetrics(
        face_count=1,
        face_area_ratio=0.2,
        face_side=300.0,
        blur_var=80.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brightness=120.0,
        width=1024,
        height=1024,
    )


def test_telemetry_event_is_deidentified() -> None:
    session = SessionState(
        session_key="s-secret",
        face_key="f-secret",
        fsm_state=FsmState.DONE,
        preset_id="demo_avatar",
        preset_version="1.0.0",
        library_version="examples",
        slots={"purpose": "a general profile avatar"},
        budget_limit=Decimal("0.50"),
        iterations=[
            Iteration(
                n=1,
                prompt_hash="h1",
                charged=True,
                cost=Decimal("0.05"),
                similarity=0.5,
                verdict=Verdict.SOFT,
            ),
            Iteration(
                n=2,
                prompt_hash="h1",
                charged=True,
                cost=Decimal("0.069"),
                similarity=0.8,
                verdict=Verdict.PASSED,
            ),
        ],
        best_result=BestResult(
            iteration_n=2,
            result_ref="sessions/s-secret/iterations/2",
            similarity=0.8,
            verdict=Verdict.PASSED,
            risk_level=RiskLevel.LOW,
        ),
    )

    with capture_logs() as captured:
        Telemetry().session_terminal(session)

    [event] = captured
    assert event["event"] == "session_terminal"
    assert event["preset_id"] == "demo_avatar"
    assert event["preset_version"] == "1.0.0"
    assert event["library_version"] == "examples"
    assert event["n_iterations"] == 2
    assert event["n_retries"] == 1
    assert event["generations_charged"] == 2
    assert event["cost"] == "0.119"  # real billed dollars, 0.05 + 0.069
    assert event["best_similarity"] == 0.8
    assert event["verdict"] == "passed"
    assert event["risk_level"] == "low"
    assert event["failure_reason"] is None

    # De-identified: nothing in the event can be joined back to a person.
    flat = repr(event)
    assert "s-secret" not in flat
    assert "f-secret" not in flat
    assert "embedding" not in flat


def test_telemetry_failure_reason_passthrough() -> None:
    session = SessionState(session_key="s1", face_key="f1", fsm_state=FsmState.CANCELLED)
    with capture_logs() as captured:
        Telemetry().session_terminal(session, failure_reason="cancelled by the caller")
    [event] = captured
    assert event["fsm_state"] == "cancelled"
    assert event["failure_reason"] == "cancelled by the caller"
    assert event["generations_charged"] == 0
    assert event["verdict"] is None
