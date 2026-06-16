"""Deterministic step planner — the no-LLM fallback behind the StepPlanner port.

One step per change, ordered from scene-level changes toward face-adjacent ones
(identity drift compounds along the chain, so the risky edits run last, on a
frame that already converged). No merging — that judgment is the LLM planner's;
this is the stable baseline the pipeline degrades to.

The plan is **not** trimmed to the budget: under greedy pay-as-you-go the runtime
reserves before each generation and stops cleanly when the next one would overdraw,
shipping whatever steps completed. A budget too small to finish the chain produces
a partial result, not a pre-emptively shortened plan.
"""

from __future__ import annotations

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
    """One change = one step, ordered scene-first → face-adjacent last."""
    ordered = sorted(analysis.changes, key=_rank)
    return [
        EditStep(n=i, title=c.target, instruction=c.instruction, targets=[c.target])
        for i, c in enumerate(ordered, start=1)
    ]


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
