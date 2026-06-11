"""Deterministic use-case classifier — the no-LLM fallback behind the port.

Maps a free-text brief onto the library's curated vocabulary by word overlap:
the token sharing the most significant words with the brief wins; nothing
overlaps → the reserved ``default`` fallback. No external calls, same input →
same output, so the pipeline degrades gracefully when the LLM classifier is
unavailable or the budget refuses it, and tests get a stable baseline.

The token-overlap idiom mirrors ``services.slot_filler._match_option`` — the two
share the same stopword-filtered word matching.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from protocols.budget import BudgetMeter
from protocols.classifier import ClassifyResult

# The reserved fall-through token (mirrors preset_matcher's reserved use_case):
# returned when no curated token overlaps the brief.
FALLBACK_USE_CASE = "default"

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "for", "i", "in", "is", "it", "me", "my", "need",
        "of", "on", "or", "photo", "picture", "the", "to", "want", "with",
    }
)  # fmt: skip

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def best_use_case(brief: str, use_cases: Sequence[str]) -> str:
    """The curated token with the largest brief overlap, else ``default``."""
    brief_tokens = _tokens(brief)
    best: str | None = None
    best_score = 0
    for token in use_cases:
        # A multi-word use_case ("formal_portrait") splits on non-word chars.
        score = len(_tokens(token.replace("_", " ")) & brief_tokens)
        if score > best_score:
            best, best_score = token, score
    return best if best is not None else FALLBACK_USE_CASE


class TokenOverlapClassifier:
    """Deterministic :class:`~protocols.classifier.UseCaseClassifier` — free."""

    async def classify(
        self, *, brief: str, use_cases: Sequence[str], meter: BudgetMeter | None = None
    ) -> ClassifyResult:
        # meter is part of the port for the LLM classifier's benefit; the
        # deterministic fallback is free and never reserves.
        return ClassifyResult(use_case=best_use_case(brief, use_cases))
