"""Telemetry — de-identified session-outcome events.

One structured event per terminal session, carrying everything needed to tune
the product (which presets converge, how many retries similarity costs, where
budgets die) and **nothing** that identifies a person: no embeddings, no
images, no ``session_key``/``face_key``. Those keys are the business service's
link to a user — telemetry must stay joinable to presets and models only, so
an exported/aggregated stream can never be walked back to who was photographed.

Slot values are included (they describe the *shoot*, not the person): they are
preset vocabulary plus the sanitized free-form scene, which prompt_builder has
already stripped of identity-touching text.

The sink is the process log under the dedicated ``telemetry`` logger — in prod
that is structured JSON the platform routes like any other metric stream; a
real metrics backend can later replace the logger behind the same call.
"""

from __future__ import annotations

from decimal import Decimal

import structlog

from schemas import SessionState

logger = structlog.get_logger("telemetry")


class Telemetry:
    """Emits the per-session outcome event."""

    def __init__(self, *, unit_price: Decimal) -> None:
        self._unit_price = unit_price

    def session_terminal(self, session: SessionState, *, failure_reason: str | None = None) -> None:
        """Record a session that reached a terminal state.

        ``failure_reason`` comes from the caller (graph failure payload,
        wall-clock timeout, cancel) — the session record itself does not
        store one.
        """
        charged = sum(1 for it in session.iterations if it.charged)
        best = session.best_result
        logger.info(
            "session_terminal",
            fsm_state=session.fsm_state.value,
            preset_id=session.preset_id,
            preset_version=session.preset_version,
            library_version=session.library_version,
            slots=dict(session.slots),
            n_iterations=len(session.iterations),
            n_retries=max(0, len(session.iterations) - 1),
            generations_charged=charged,
            budget_limit=session.budget_limit,
            cost=str(self._unit_price * charged),
            best_similarity=best.similarity if best else None,
            verdict=best.verdict.value if best else None,
            risk_level=best.risk_level.value if best else None,
            failure_reason=failure_reason,
        )
