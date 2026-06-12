"""Use-case token overlap — the deterministic engine behind the brief parser.

Maps a free-text brief onto the library's curated vocabulary by word overlap:
the token sharing the most significant words with the brief wins; nothing
overlaps → the reserved ``default`` fallback. No external calls, same input →
same output. The :class:`~services.brief_parser.DeterministicBriefParser` builds
on this for its no-LLM fallback, and ``api/routes`` reuses the reserved token.

The token-overlap idiom mirrors ``services.slot_filler._match_option`` — the two
share the same stopword-filtered word matching.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

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
