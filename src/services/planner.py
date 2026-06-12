"""Deterministic step planner — the no-LLM fallback behind the StepPlanner port,
plus the budget-trim that keeps a plan honest.

The deterministic plan is the simplest faithful one: a generate-mode brief is a
single step; an edit-mode brief is one step per change. No merging, no splitting
— that judgment is the LLM planner's; this is the stable baseline the pipeline
degrades to.

:func:`fit_to_budget` is separate and always deterministic: when the budget
cannot fund every step it trims the **tail** (marks the trailing steps
``skipped``) and returns a note, so a partial plan is explicit on the record
rather than a silent cap.
"""

from __future__ import annotations

from collections.abc import Sequence

from protocols.budget import BudgetMeter
from protocols.planner import PlanResult
from schemas import BriefAnalysis, EditStep


def deterministic_steps(analysis: BriefAnalysis) -> list[EditStep]:
    """One change = one step (edit); a single step (generate)."""
    if analysis.mode == "generate":
        instruction = "; ".join(c.instruction for c in analysis.changes)
        if not instruction:
            instruction = (
                f"a {analysis.use_case} photo" if analysis.use_case else "a styled portrait"
            )
        return [
            EditStep(
                n=1,
                title=analysis.use_case or "generate",
                instruction=instruction,
                targets=[c.target for c in analysis.changes],
            )
        ]
    return [
        EditStep(n=i, title=c.target, instruction=c.instruction, targets=[c.target])
        for i, c in enumerate(analysis.changes, start=1)
    ]


def fit_to_budget(steps: Sequence[EditStep], max_steps: int) -> tuple[list[EditStep], str | None]:
    """Keep the first ``max_steps`` steps; mark the trailing ones ``skipped``.

    Returns the (re-numbered-unchanged) steps and a note when anything was
    trimmed — never a silent drop. ``max_steps <= 0`` skips everything, which the
    caller reads as "budget cannot fund even one step".
    """
    if max_steps >= len(steps):
        return list(steps), None
    kept = [
        step if position <= max_steps else step.model_copy(update={"status": "skipped"})
        for position, step in enumerate(steps, start=1)
    ]
    note = f"budget funds {max(max_steps, 0)} of {len(steps)} steps; the rest are skipped"
    return kept, note


class DeterministicStepPlanner:
    """Free :class:`~protocols.planner.StepPlanner` — one change = one step, no calls."""

    async def plan(
        self, *, analysis: BriefAnalysis, meter: BudgetMeter | None = None
    ) -> PlanResult:
        # meter is part of the port for the LLM planner's benefit; the
        # deterministic fallback is free and never reserves.
        return PlanResult(steps=deterministic_steps(analysis))
