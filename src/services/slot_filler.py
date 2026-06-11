"""Deterministic slot filler — the no-LLM fallback behind the SlotFiller port.

Maps the user's answer onto the preset's **own vocabulary** by word overlap:
the enum option of the asked slot sharing the most significant words with the
answer wins; nothing overlaps → the preset's defaults. No external calls, same
input → same output, so the pipeline degrades gracefully when the LLM filler
(a later step) is unavailable, and tests get a stable baseline.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from protocols.budget import BudgetMeter
from protocols.slot_filler import SlotFill
from schemas import FrameMetrics, Preset

# Words too common to signal a choice — keeps "a photo for my profile" from
# matching an option on its articles and filler words alone.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "for", "i", "in", "is", "it", "me", "my", "need",
        "of", "on", "or", "photo", "picture", "the", "to", "want", "with",
    }
)  # fmt: skip

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def _match_option(answer: str, options: list[object]) -> str | None:
    """Enum option with the largest token overlap; ties keep the earlier option."""
    answer_tokens = _tokens(answer)
    best: str | None = None
    best_score = 0
    for option in options:
        score = len(_tokens(str(option)) & answer_tokens)
        if score > best_score:
            best, best_score = str(option), score
    return best


class DefaultSlotFiller:
    """Fills every slot from its default; the asked slot from the user's answer —
    snapped to the enum when the slot has one, taken verbatim when it is free-form
    (the fallback preset's ``scene``). The prompt builder sanitizes free-form text."""

    async def fill(
        self,
        *,
        preset: Preset,
        user_answer: str | None,
        photo_analysis: FrameMetrics | None,
        meter: BudgetMeter | None = None,
    ) -> SlotFill:
        # photo_analysis and meter are part of the port for the LLM filler's
        # benefit; the deterministic fallback is free and uses neither.
        slots: dict[str, str] = {}
        for name, slot in preset.slots.items():
            value = slot.default
            if slot.ask and user_answer is not None:
                if slot.enum:
                    # Enum ask slot: snap the answer to the preset's vocabulary.
                    value = _match_option(user_answer, slot.enum) or value
                else:
                    # Free-form ask slot (e.g. the fallback's `scene`): the
                    # user's own words fill it. The prompt builder is the trust
                    # boundary that sanitizes this text before it enters the
                    # frozen blocks, so the filler passes it through verbatim.
                    value = user_answer
            if value is None:
                raise ValueError(
                    f"slot {name!r} of preset {preset.id!r} has no default and no match"
                )
            slots[name] = str(value)
        return SlotFill(slots=slots, addendum="")


def apply_composition(
    preset: Preset, slots: Mapping[str, str], composition_id: str
) -> dict[str, str]:
    """Overlay the chosen composition's ``slot_overrides`` onto resolved slots.

    Pure merge — the preset library tests guarantee overrides reference declared
    slots, and the prompt builder re-validates values, so this stays a dumb,
    deterministic overlay. An unknown composition id fails loudly.
    """
    composition = next((c for c in preset.compositions if c.id == composition_id), None)
    if composition is None:
        raise ValueError(f"preset {preset.id!r} has no composition {composition_id!r}")
    merged = dict(slots)
    for name, value in composition.slot_overrides.items():
        merged[name] = str(value)
    return merged
