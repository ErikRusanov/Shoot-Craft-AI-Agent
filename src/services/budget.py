"""Budget — reserving paid generation slots.

``budget_limit`` is the number of *paid generations* the business service
granted this session. The only thing that consumes a slot is an actual call to
the image generator: reserve **immediately before** invoking it, and never for
retries that fail earlier in the pipeline (gate rejections, prompt building,
face-check of an already-generated frame are all free).

The reservation itself is a single atomic check-and-increment on the store
(Lua in Redis), so concurrent loop iterations can never overdraw — exactly
``limit`` reservations succeed, no matter the interleaving. There are no
refunds: a generation that was started is a generation that was paid for.
"""

from __future__ import annotations

from protocols import StateStore


class BudgetService:
    """Hands out paid-generation slots, atomically, never past the limit."""

    def __init__(self, store: StateStore) -> None:
        self._store = store

    async def reserve_generation(self, session_key: str, *, limit: int, ttl_seconds: int) -> bool:
        """Reserve one paid generation; ``False`` means the budget is exhausted.

        Call this right before the generator call it pays for — a ``True``
        return is already charged.
        """
        return await self._store.check_and_incr_budget(
            session_key, limit=limit, ttl_seconds=ttl_seconds
        )
