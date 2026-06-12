"""Deterministic prompt writer — the no-LLM fallback behind the PromptWriter port.

Reproduces today's behavior exactly so the pipeline degrades gracefully when the
writer LLM is unavailable or the budget refuses it:

- :meth:`compose` returns the request's ``template_body`` — the filled
  ``prompt_structure`` the caller prepared — verbatim;
- :meth:`revise` appends the one sanctioned identity-emphasis line, the same
  fixed wording the generation loop used to glue on for retries (idempotent, so
  a third attempt does not stack it twice).

No external calls, same input → same output, and tests get a stable baseline.
"""

from __future__ import annotations

from protocols.budget import BudgetMeter
from protocols.prompt_writer import WriteRequest, WriteResult, WriterFeedback
from schemas import FrameMetrics

# The one sanctioned retry text: appended after the body, never edited into it.
# Fixed wording keeps retries reproducible. (Lifted from the generation loop,
# where it was glued on as an addendum; the writer owns it now.)
IDENTITY_EMPHASIS = (
    "Critical: render the exact same person as in the reference photo — identical "
    "facial geometry, eyes, nose, lips, jawline, skin tone and texture. "
    "A faithful, recognizable likeness matters more than any stylistic choice."
)


def emphasize(body: str) -> str:
    """Append the identity-emphasis line once; idempotent across retries."""
    if IDENTITY_EMPHASIS in body:
        return body
    return f"{body.strip()}\n\n{IDENTITY_EMPHASIS}"


class DeterministicPromptWriter:
    """Free :class:`~protocols.prompt_writer.PromptWriter` — the template, no LLM."""

    async def compose(
        self,
        request: WriteRequest,
        *,
        photo_metrics: FrameMetrics | None = None,
        meter: BudgetMeter | None = None,
    ) -> WriteResult:
        # photo_metrics and meter are part of the port for the LLM writer's
        # benefit; the deterministic fallback is free and uses neither.
        return WriteResult(body=request.template_body)

    async def revise(
        self,
        prev_body: str,
        feedback: WriterFeedback,
        *,
        request: WriteRequest,
        meter: BudgetMeter | None = None,
    ) -> WriteResult:
        return WriteResult(body=emphasize(prev_body))
