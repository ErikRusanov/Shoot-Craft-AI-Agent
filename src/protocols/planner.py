"""Port: step planner — cut a brief's changes into an ordered step plan.

The generator is reference-conditioned edit and chains naturally, so a complex
brief (background + lighting + t-shirt + microphone) decomposes into steps the
loop runs in sequence, each step's kept-best feeding the next. Identity drift
compounds with every pass, so a step is one change by default (only same-region
scene tweaks merge) and the order runs from scene-level changes toward the
face-adjacent ones. A generate-mode brief is the degenerate case — a single
step.

Failure policy mirrors the other LLM ports: reserve through the
:class:`~protocols.budget.BudgetMeter`, and on any misbehavior degrade to the
deterministic "one change = one step" plan rather than failing the session.
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple, Protocol, runtime_checkable

from protocols.budget import BudgetMeter
from schemas import BriefAnalysis, EditStep, PhotoInventory, ProviderUsage


class PlanResult(NamedTuple):
    """The ordered steps plus what the call billed (0 for the deterministic plan)."""

    steps: list[EditStep]
    usage: ProviderUsage | None = None
    cost: Decimal = Decimal("0")


@runtime_checkable
class StepPlanner(Protocol):
    """Cut a :class:`BriefAnalysis` into an ordered list of :class:`EditStep`."""

    async def plan(
        self,
        *,
        analysis: BriefAnalysis,
        inventory: PhotoInventory | None = None,
        meter: BudgetMeter | None = None,
    ) -> PlanResult:
        """Decompose ``analysis.changes`` into ordered steps.

        ``inventory`` (when extracted) lets the planner resolve references
        against what the photo actually shows ("replace the existing earbud…"
        rather than "add a headset"). ``meter`` is the session budget for the
        paid path; a refused budget degrades to the deterministic plan. Every
        change must be covered; nothing is dropped silently (budget trimming is
        a separate, explicit step).
        """
        ...
