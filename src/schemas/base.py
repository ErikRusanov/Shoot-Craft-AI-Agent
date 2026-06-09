"""Shared pydantic bases for the contract.

Two bases, one rule: reject unknown keys so a malformed payload (a typo, a
stale client) fails loudly instead of silently dropping data. Aggregate roots
that are versioned on the wire add ``schema_v``; pure value objects nested inside
them do not (the root's version covers the whole tree — same convention as
``schemas/presets.py``, where only ``Preset`` carries ``schema_v``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Reject unknown keys. Base for nested value objects (no own version)."""

    model_config = ConfigDict(extra="forbid")


class SchemaModel(StrictModel):
    """A versioned aggregate root: strict, plus a contract version.

    ``schema_v`` versions *this shape*. Bump it only on a breaking change to the
    model so a reader can branch on the value rather than guess.
    """

    schema_v: int = 1
