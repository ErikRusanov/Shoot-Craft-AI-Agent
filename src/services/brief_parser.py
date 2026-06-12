"""Deterministic brief parser — the no-LLM fallback behind the BriefParser port.

This is exactly today's classifier behavior, lifted into the new shape so the
pipeline degrades gracefully when the LLM parser is unavailable or the budget
refuses it, and tests get a stable baseline:

- a caller-supplied ``use_case`` (the business service knows the target) →
  ``mode=generate`` with that token, no deltas;
- otherwise token-overlap over the curated vocabulary: a curated token wins →
  ``mode=generate`` with it; nothing overlaps → ``mode=edit`` carrying the whole
  brief as a single change (or no change at all when the brief is empty, which
  the ask node reads as "nothing actionable").

It cannot reason about a preserve-list or split deltas — that is the LLM
parser's job. The deterministic parse keeps ``preserve`` empty and never
invents conflicts; the edit framing freezes the face regardless.
"""

from __future__ import annotations

from collections.abc import Sequence

from protocols.brief_parser import ParseResult
from protocols.budget import BudgetMeter
from schemas import BriefAnalysis, Change
from services.classifier import FALLBACK_USE_CASE, best_use_case

# The reserved target name for the single free-form change of an edit-mode
# brief — aligns with the fallback preset's free-form ``scene`` slot.
SCENE_TARGET = "scene"


def deterministic_analysis(
    brief: str, use_case: str | None, use_cases: Sequence[str]
) -> BriefAnalysis:
    """Today's classifier behavior, expressed as a :class:`BriefAnalysis`."""
    if use_case:
        # The caller knows the target — a curated, target-driven generate.
        return BriefAnalysis(mode="generate", use_case=use_case)

    token = best_use_case(brief, use_cases)
    if token != FALLBACK_USE_CASE:
        return BriefAnalysis(mode="generate", use_case=token)

    # Nothing matched a curated preset → an edit of the user's own photo. The
    # whole brief becomes one change; an empty brief leaves no change, so the ask
    # node knows to ask for one.
    changes = [Change(target=SCENE_TARGET, instruction=brief.strip())] if brief.strip() else []
    return BriefAnalysis(mode="edit", use_case=None, changes=changes)


class DeterministicBriefParser:
    """Free :class:`~protocols.brief_parser.BriefParser` — token overlap, no calls."""

    async def parse(
        self,
        *,
        brief: str,
        use_case: str | None,
        use_cases: Sequence[str],
        meter: BudgetMeter | None = None,
    ) -> ParseResult:
        # meter is part of the port for the LLM parser's benefit; the
        # deterministic fallback is free and never reserves.
        return ParseResult(analysis=deterministic_analysis(brief, use_case, use_cases))
