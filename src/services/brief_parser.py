"""Deterministic brief parser — the no-LLM fallback behind the BriefParser port.

The pipeline always edits the user's reference photo. When the brief is
non-empty the whole text becomes a single scene change; an empty brief leaves no
change at all, which the ask node reads as "nothing actionable" and prompts the
user for a scene description.

It cannot reason about a preserve-list or split deltas — that is the LLM
parser's job. The deterministic parse keeps ``preserve`` empty and never
invents conflicts; the edit framing freezes the face regardless.
"""

from __future__ import annotations

from protocols.brief_parser import ParseResult
from protocols.budget import BudgetMeter
from schemas import BriefAnalysis, Change

# The reserved target name for the single free-form change of an edit-mode
# brief — aligns with the fallback preset's free-form ``scene`` slot.
SCENE_TARGET = "scene"


def deterministic_analysis(brief: str) -> BriefAnalysis:
    """Always edit mode: the brief becomes one scene change (or no change)."""
    changes = [Change(target=SCENE_TARGET, instruction=brief.strip())] if brief.strip() else []
    return BriefAnalysis(changes=changes)


class DeterministicBriefParser:
    """Free :class:`~protocols.brief_parser.BriefParser` — no calls."""

    async def parse(
        self,
        *,
        brief: str,
        meter: BudgetMeter | None = None,
    ) -> ParseResult:
        # meter is part of the port for the LLM parser's benefit; the
        # deterministic fallback is free and never reserves.
        return ParseResult(analysis=deterministic_analysis(brief))
