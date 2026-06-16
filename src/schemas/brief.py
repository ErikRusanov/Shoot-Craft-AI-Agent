"""Brief analysis — the structured reading of the user's request.

Replaces the single use-case token the classifier used to collapse a brief into.
A brief carries two things that token threw away: what must be **preserved** (the
user said "keep X as is") and what must **change** (the deltas, "make Y blue").
Those drive the whole pipeline — preset constraints, the step plan — so they are
first-class state, not re-derived downstream.

The pipeline is always **edit-mode**: start from the user's photo, apply only the
named changes, keep everything else. ``use_case`` is retained as an informational
field for tracing; it is never operational.

``conflicts`` are asks that contradict a locked preset attribute or try to edit
the face/identity; they are surfaced to the user, **never** silently dropped.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from schemas.base import SchemaModel, StrictModel


class Change(StrictModel):
    """One requested delta: change ``target`` per ``instruction``.

    ``target`` is a short slot-like noun the planner groups steps by (background,
    lighting, clothing); ``instruction`` is the user's intent in their own words.
    """

    target: str
    instruction: str


class BriefAnalysis(SchemaModel):
    """The structured reading of one brief — the parser's output.

    Kept on the session so a delivered result can be explained from what was
    actually understood, not from the raw text after the fact.
    """

    schema_v: int = 1
    mode: Literal["edit"] = "edit"
    use_case: str | None = None  # informational/tracing only, not operational
    # What stays put — face is always implied, but the user may pin pose,
    # framing, clothing, setting. Surfaced to the writer as the preserve-list.
    preserve: list[str] = Field(default_factory=list)
    changes: list[Change] = Field(default_factory=list)
    # Asks that contradict a locked attribute or try to edit the face/identity.
    conflicts: list[str] = Field(default_factory=list)
