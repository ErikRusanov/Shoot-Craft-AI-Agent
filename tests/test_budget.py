"""BudgetService meter: reserve → settle/cancel against a per-session dollar limit.

Atomicity itself is pinned by the store contract and tests/test_concurrency.py;
here we pin the meter surface — reservations never exceed the limit, settle moves
the counter to the real cost, cancel refunds in full, and a reservation closes
exactly once.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from schemas import PaidCallKind
from services.budget import BudgetService
from services.connectors import InMemoryStateStore
from services.pricing import PricingTable

TTL = 60
GEN = "gen-model"
LITE = "lite-model"


def _pricing() -> PricingTable:
    return PricingTable.default(generation_model=GEN, lite_model=LITE)


def _service() -> tuple[BudgetService, InMemoryStateStore]:
    store = InMemoryStateStore()
    return BudgetService(store, _pricing()), store


async def test_reserve_until_limit_then_refuses() -> None:
    svc, _ = _service()
    # Flat SLOT_FILL estimate is $0.002; a $0.005 limit covers two, not three.
    meter = svc.meter("sess-1", limit=Decimal("0.005"), ttl_seconds=TTL)
    first = await meter.reserve(PaidCallKind.SLOT_FILL)
    second = await meter.reserve(PaidCallKind.SLOT_FILL)
    third = await meter.reserve(PaidCallKind.SLOT_FILL)
    assert first is not None and second is not None
    assert third is None  # 0.006 > 0.005


async def test_explicit_estimate_overrides_the_flat_default() -> None:
    svc, _ = _service()
    meter = svc.meter("sess-1", limit=Decimal("0.10"), ttl_seconds=TTL)
    reservation = await meter.reserve(PaidCallKind.GENERATION, estimate=Decimal("0.08"))
    assert reservation is not None
    # 0.08 reserved; a second 0.08 would exceed 0.10.
    assert await meter.reserve(PaidCallKind.GENERATION, estimate=Decimal("0.08")) is None


async def test_settle_frees_the_overestimate_for_a_later_call() -> None:
    svc, _ = _service()
    meter = svc.meter("sess-1", limit=Decimal("0.10"), ttl_seconds=TTL)
    reservation = await meter.reserve(PaidCallKind.GENERATION, estimate=Decimal("0.08"))
    assert reservation is not None
    # Real cost was only 0.05 — settling refunds the 0.03 overestimate...
    assert await reservation.settle(Decimal("0.05")) == Decimal("0.05")
    # ...so a 0.05 reservation now fits under the limit (0.05 + 0.05 == 0.10).
    assert await meter.reserve(PaidCallKind.GENERATION, estimate=Decimal("0.05")) is not None


async def test_settle_none_keeps_the_reserved_estimate() -> None:
    svc, _ = _service()
    meter = svc.meter("sess-1", limit=Decimal("0.10"), ttl_seconds=TTL)
    reservation = await meter.reserve(PaidCallKind.GENERATION, estimate=Decimal("0.06"))
    assert reservation is not None
    # Provider reported no cost: keep the held estimate as the spend.
    assert await reservation.settle(None) == Decimal("0.06")
    assert await meter.reserve(PaidCallKind.GENERATION, estimate=Decimal("0.05")) is None


async def test_cancel_refunds_in_full() -> None:
    svc, _ = _service()
    meter = svc.meter("sess-1", limit=Decimal("0.08"), ttl_seconds=TTL)
    reservation = await meter.reserve(PaidCallKind.GENERATION, estimate=Decimal("0.08"))
    assert reservation is not None
    await reservation.cancel()
    # The whole estimate is back — a fresh full reservation fits again.
    assert await meter.reserve(PaidCallKind.GENERATION, estimate=Decimal("0.08")) is not None


async def test_double_close_raises() -> None:
    svc, _ = _service()
    meter = svc.meter("sess-1", limit=Decimal("1"), ttl_seconds=TTL)
    reservation = await meter.reserve(PaidCallKind.SLOT_FILL)
    assert reservation is not None
    await reservation.settle(Decimal("0.001"))
    with pytest.raises(RuntimeError):
        await reservation.settle(Decimal("0.001"))
    with pytest.raises(RuntimeError):
        await reservation.cancel()


async def test_sessions_do_not_share_budget() -> None:
    svc, _ = _service()
    one = svc.meter("sess-1", limit=Decimal("0.002"), ttl_seconds=TTL)
    two = svc.meter("sess-2", limit=Decimal("0.002"), ttl_seconds=TTL)
    assert await one.reserve(PaidCallKind.SLOT_FILL) is not None
    assert await one.reserve(PaidCallKind.SLOT_FILL) is None
    assert await two.reserve(PaidCallKind.SLOT_FILL) is not None
