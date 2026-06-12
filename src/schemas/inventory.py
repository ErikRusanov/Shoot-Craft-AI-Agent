"""Photo inventory — what is *visible* in the reference photo, in words.

A textual catalogue of the person and scene (pose, hands, accessories, clothing,
hair, lighting, background) extracted once per reference photo by a VLM. It
exists so edit-mode prompts can enumerate concrete untouchables ("wedding ring
on the right hand", "white earbud in the left ear") instead of a generic "keep
everything" — the single biggest lever for identity preservation in chained
edits.

Appearance text, not biometrics: no facial geometry, no identity vector. It is
stored TTL-bound on the :class:`~schemas.state.FaceProfile` and may appear in
recorded prompt text (prompts describe appearance by nature) but is never
logged.
"""

from __future__ import annotations

from pydantic import Field

from schemas.base import SchemaModel


class PhotoInventory(SchemaModel):
    """Visible-attribute catalogue of one reference photo.

    Every field is free text from the extractor; empty means "not visible /
    not extracted". An entirely empty inventory is the deterministic fallback —
    downstream consumers degrade to generic preserve phrasing.
    """

    schema_v: int = 1
    pose: str = ""  # body orientation, posture, arm and hand placement
    hands: str = ""  # visibility, gesture, held objects
    # One item per visible accessory, with placement ("wedding ring on the
    # right hand") so a step editing one accessory can unlock just that item.
    accessories: list[str] = Field(default_factory=list)
    clothing: str = ""  # garment, color, fit, neckline
    hair: str = ""
    facial_hair: str = ""
    framing: str = ""  # crop, camera distance and angle
    lighting: str = ""  # light direction, quality, overall color grade
    background: str = ""  # one-sentence summary

    def is_empty(self) -> bool:
        """True when nothing was extracted — consumers fall back to generic text."""
        return not any(
            [
                self.pose,
                self.hands,
                self.accessories,
                self.clothing,
                self.hair,
                self.facial_hair,
                self.framing,
                self.lighting,
                self.background,
            ]
        )
