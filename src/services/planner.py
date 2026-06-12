"""Deterministic step planner — the no-LLM fallback behind the StepPlanner port,
plus the budget-trim that keeps a plan honest.

The deterministic plan is the simplest faithful one: a generate-mode brief is a
single step; an edit-mode brief is one step per change, ordered from scene-level
changes toward face-adjacent ones (identity drift compounds along the chain, so
the risky edits run last, on a frame that already converged). No merging — that
judgment is the LLM planner's; this is the stable baseline the pipeline
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
from schemas import BriefAnalysis, Change, EditStep, PhotoInventory

# Identity-sensitivity ranking for the deterministic order: scene-level first
# (0), body/clothing next (1), accessories and face-adjacent items last (2).
# Unknown targets rank with clothing (1) — riskier than the scene, safer than
# the face. The sort is stable, so the brief's order survives within a rank.
_REGION_RANK: tuple[tuple[frozenset[str], int], ...] = (
    (
        frozenset(
            {
                "background",
                "backdrop",
                "setting",
                "scene",
                "environment",
                "lighting",
                "light",
                "grade",
                "color",
                "colour",
                "tone",
            }
        ),
        0,
    ),
    (
        frozenset(
            {
                "accessory",
                "accessories",
                "glasses",
                "sunglasses",
                "earring",
                "earrings",
                "earbud",
                "headset",
                "hat",
                "cap",
                "mic",
                "microphone",
                "jewelry",
                "jewellery",
                "necklace",
                "headphones",
            }
        ),
        2,
    ),
)
_DEFAULT_RANK = 1


def _rank(change: Change) -> int:
    tokens = set(change.target.lower().replace("_", " ").split())
    for vocabulary, rank in _REGION_RANK:
        if tokens & vocabulary:
            return rank
    return _DEFAULT_RANK


def deterministic_steps(analysis: BriefAnalysis) -> list[EditStep]:
    """One change = one step (edit, scene-first order); a single step (generate)."""
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
    ordered = sorted(analysis.changes, key=_rank)
    return [
        EditStep(n=i, title=c.target, instruction=c.instruction, targets=[c.target])
        for i, c in enumerate(ordered, start=1)
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
        self,
        *,
        analysis: BriefAnalysis,
        inventory: PhotoInventory | None = None,
        meter: BudgetMeter | None = None,
    ) -> PlanResult:
        # inventory/meter are part of the port for the LLM planner's benefit;
        # the deterministic fallback is free and resolves nothing against the photo.
        return PlanResult(steps=deterministic_steps(analysis))
