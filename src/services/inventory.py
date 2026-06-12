"""Deterministic no-LLM fallback for the inventory extractor port.

An empty inventory is a valid one: the edit-mode lock block then renders only
the generic person lock instead of a concrete enumeration — degraded, never
broken. This keeps the pipeline's invariant that no LLM stage can fail a
session.
"""

from __future__ import annotations

from protocols.budget import BudgetMeter
from protocols.inventory import InventoryResult
from schemas import PhotoInventory


class EmptyInventoryExtractor:
    """The free fallback: no call, no spend, an empty inventory."""

    async def extract(
        self,
        image: bytes,
        *,
        meter: BudgetMeter | None = None,
    ) -> InventoryResult:
        return InventoryResult(inventory=PhotoInventory())
