"""Port: prompt writer — composes the *body* of a generation prompt.

The agent's core. Instead of gluing a prompt from preset template pieces, an LLM
composes the scene/edit body per situation: the mode (edit vs generate), the
step's instruction, what to preserve, the locked attribute values
(informational), the preset's style notes, and the input-photo metrics. On a
retry it *revises* that body against the prior attempt's face-check feedback.

The writer's authority is deliberately narrow, and that narrowness **is** the
port's invariant: it produces the **body only**. ``identity_instruction``,
``negative_prompt`` and the locked attributes are frozen content it never sees as
editable text — :mod:`services.prompt_builder` assembles them deterministically
around the body, and sanitizes the body before it lands. So even a fully
compromised writer cannot touch identity, the exclusions, or a lock.

Failure policy mirrors the other LLM ports: reserve a dollar slot through the
:class:`~protocols.budget.BudgetMeter`, and on any misbehavior (budget refusal,
transport failure, unparseable output) degrade to the deterministic writer — the
filled ``prompt_structure`` template — rather than failing the session. The
deterministic ``revise`` is the old fixed identity-emphasis addendum.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, NamedTuple, Protocol, runtime_checkable

from protocols.budget import BudgetMeter
from schemas import FrameMetrics, PhotoInventory, ProviderUsage, Verdict


class WriteRequest(NamedTuple):
    """Everything the writer needs to compose one step's body — and nothing more.

    Carries no frozen secret: ``identity_instruction`` and ``negative_prompt``
    are never here. ``template_body`` is the deterministic fallback (the filled
    ``prompt_structure``) the writer returns verbatim when it cannot do better;
    ``locked`` lists non-negotiable attribute values so the body does not fight
    them, and ``defaults`` the preset defaults the body should honor unless a
    change overrides them. ``inventory`` (edit mode) is what the reference photo
    shows, so the body can integrate the change concretely; ``applied`` are the
    earlier steps' results at their new values, already locked by the builder —
    informational, so the body does not re-describe or fight them.
    """

    mode: Literal["edit", "generate"]
    instruction: str  # the step's directive: what to do this step, in words
    preserve: list[str]  # what to keep exactly as the reference shows it
    locked: dict[str, str]  # locked attribute → fixed value (informational)
    defaults: dict[str, str]  # non-locked preset defaults to honor unless changed
    style_notes: str
    template_body: str  # deterministic fallback body (filled prompt_structure)
    inventory: PhotoInventory | None = None  # what the reference photo shows
    applied: tuple[str, ...] = ()  # earlier steps' changes at their new values


class WriterFeedback(NamedTuple):
    """What the prior attempt measured — drives a revision.

    Face-check only for now (similarity + verdict + which attempt produced it);
    the VLM compliance critique plugs in here later without re-architecture.
    """

    similarity: float | None
    verdict: Verdict | None
    attempt: int | None = None  # 1-based attempt number that was measured


class WriteResult(NamedTuple):
    """The composed body and what the call billed (0 for the deterministic writer)."""

    body: str
    usage: ProviderUsage | None = None
    cost: Decimal = Decimal("0")


@runtime_checkable
class PromptWriter(Protocol):
    """Compose (and revise) the body of a generation prompt."""

    async def compose(
        self,
        request: WriteRequest,
        *,
        photo_metrics: FrameMetrics | None = None,
        meter: BudgetMeter | None = None,
    ) -> WriteResult:
        """Write the scene/edit body for ``request`` — body text only.

        ``meter`` is the session budget: reserve before the paid call and settle
        after; a refused budget (or ``None``) degrades to ``request.template_body``.
        """
        ...

    async def revise(
        self,
        prev_body: str,
        feedback: WriterFeedback,
        *,
        request: WriteRequest,
        meter: BudgetMeter | None = None,
    ) -> WriteResult:
        """Revise ``prev_body`` against ``feedback`` to strengthen the next attempt.

        The deterministic fallback appends the fixed identity-emphasis line; an
        LLM writer re-composes with the feedback in view. Body text only.
        """
        ...
