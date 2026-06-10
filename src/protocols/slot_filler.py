"""Port: slot filler — resolves a preset's slot values for the prompt.

The filler's authority is deliberately narrow, and that narrowness **is** the
port's invariant: an implementation produces slot *values* (drawn from the
preset's own vocabulary) plus a short free-text addendum — nothing else.
``identity_instruction``, ``prompt_structure`` and ``negative_prompt`` are
frozen preset content the filler never edits or extends, including on retry.
An LLM-backed implementation works under exactly the same contract, and the
prompt builder re-validates every value against the preset, so a misbehaving
filler fails loudly instead of leaking into the frozen blocks.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

from schemas import FrameMetrics, Preset


class SlotFill(NamedTuple):
    """Resolved slot values plus the one sanctioned free-text extension."""

    slots: dict[str, str]
    # Appended after the prompt structure, never inside it; empty means none.
    addendum: str = ""


@runtime_checkable
class SlotFiller(Protocol):
    """Resolve every slot declared by a preset to a concrete value."""

    async def fill(
        self,
        *,
        preset: Preset,
        user_answer: str | None,
        photo_analysis: FrameMetrics | None,
    ) -> SlotFill:
        """Return a value for **every** slot in ``preset.slots``.

        ``user_answer`` is the reply to the preset's single ``ask:true``
        question (``None`` when the user has not answered); ``photo_analysis``
        carries the measured input-frame metrics when available. A slot that
        declares an ``enum`` must resolve to one of its members.
        """
        ...
