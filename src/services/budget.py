"""Budget — reserve a dollar slot, settle it to the call's real cost.

``budget_limit`` is a per-session **dollar** ceiling (pay-as-you-go), and it
covers every paid call: image generations and the auxiliary LLM calls (slot
fill, use-case classify). The only way to spend is to reserve first — a
conservative estimate is added to the per-session counter atomically (Lua in
Redis), so concurrent calls can never both slip past the last dollar — then
settle to the real ``usage.cost`` afterwards.

Strictness is deliberate: overspend is forbidden (a refused reservation ends the
work cleanly), because under pay-as-you-go an overdraw is money we cannot
collect. The price is a small "stuck" remnant — the gap between a padded
reservation and the cheaper real cost — that may sit unusable below the limit
until the session expires; it is visible to the business service as ``cost_spent``
and is far cheaper than an uncollectable overdraw.

The counter lives in micro-USD inside the store (Redis cannot increment a
Decimal); this service is where USD meets that integer grid via :mod:`utils.money`.
"""

from __future__ import annotations

from decimal import Decimal

from protocols import StateStore
from protocols.budget import BudgetMeter, BudgetReservation
from schemas.enums import PaidCallKind
from services.pricing import PricingTable
from utils.money import to_micro


class _Reservation(BudgetReservation):
    """A single held reservation against one session's micro-USD counter."""

    def __init__(
        self, store: StateStore, session_key: str, *, estimate: Decimal, ttl_seconds: int
    ) -> None:
        self._store = store
        self._session_key = session_key
        self._estimate = estimate
        self._estimate_micro = to_micro(estimate)
        self._ttl = ttl_seconds
        self._closed = False

    async def settle(self, actual_cost: Decimal | None = None) -> Decimal:
        self._close()
        if actual_cost is None:
            # Provider charged but reported no cost: keep the held estimate.
            return self._estimate
        delta = to_micro(actual_cost) - self._estimate_micro
        if delta != 0:
            await self._store.budget_adjust(
                self._session_key, delta_micro=delta, ttl_seconds=self._ttl
            )
        return actual_cost

    async def cancel(self) -> None:
        self._close()
        await self._store.budget_adjust(
            self._session_key, delta_micro=-self._estimate_micro, ttl_seconds=self._ttl
        )

    def _close(self) -> None:
        if self._closed:
            raise RuntimeError("budget reservation already settled or cancelled")
        self._closed = True


class _Meter(BudgetMeter):
    """Per-session meter: prices a reservation, reserves it atomically, or refuses."""

    def __init__(
        self,
        store: StateStore,
        pricing: PricingTable,
        session_key: str,
        *,
        limit: Decimal,
        ttl_seconds: int,
    ) -> None:
        self._store = store
        self._pricing = pricing
        self._session_key = session_key
        self._limit_micro = to_micro(limit)
        self._ttl = ttl_seconds

    async def reserve(
        self, kind: PaidCallKind, *, estimate: Decimal | None = None
    ) -> _Reservation | None:
        amount = estimate if estimate is not None else self._pricing.flat_estimate(kind)
        reserved = await self._store.budget_reserve(
            self._session_key,
            estimate_micro=to_micro(amount),
            limit_micro=self._limit_micro,
            ttl_seconds=self._ttl,
        )
        if not reserved:
            return None
        return _Reservation(self._store, self._session_key, estimate=amount, ttl_seconds=self._ttl)


class BudgetService:
    """Hands out per-session :class:`~protocols.budget.BudgetMeter`s."""

    def __init__(self, store: StateStore, pricing: PricingTable) -> None:
        self._store = store
        self._pricing = pricing

    def meter(self, session_key: str, *, limit: Decimal, ttl_seconds: int) -> _Meter:
        """A meter scoped to one session and its dollar ``limit``."""
        return _Meter(self._store, self._pricing, session_key, limit=limit, ttl_seconds=ttl_seconds)
