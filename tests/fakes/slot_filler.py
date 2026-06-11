"""Slot filler fake — preconfigured answers, no matching logic.

Returns the configured :class:`SlotFill` verbatim (or the preset's defaults
when none is given) and records every call, so tests can both steer what the
pipeline sees and assert what it asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

from protocols.budget import BudgetMeter
from protocols.slot_filler import SlotFill
from schemas import FrameMetrics, Preset


@dataclass
class FillCall:
    """One recorded :meth:`FixedSlotFiller.fill` invocation."""

    preset_id: str
    user_answer: str | None
    photo_analysis: FrameMetrics | None


class FixedSlotFiller:
    """Returns ``fill`` for every call; preset defaults when unconfigured."""

    def __init__(self, fill: SlotFill | None = None) -> None:
        self._fill = fill
        self.calls: list[FillCall] = []

    async def fill(
        self,
        *,
        preset: Preset,
        user_answer: str | None,
        photo_analysis: FrameMetrics | None,
        meter: BudgetMeter | None = None,
    ) -> SlotFill:
        self.calls.append(FillCall(preset.id, user_answer, photo_analysis))
        if self._fill is not None:
            return self._fill
        # A free-form ask slot (no enum, no default) has no value unless the user
        # supplied one — mirror DefaultSlotFiller and take the answer verbatim.
        slots: dict[str, str] = {}
        for name, slot in preset.slots.items():
            if slot.default is not None:
                slots[name] = str(slot.default)
            elif slot.ask and user_answer is not None:
                slots[name] = user_answer
        return SlotFill(slots=slots)
