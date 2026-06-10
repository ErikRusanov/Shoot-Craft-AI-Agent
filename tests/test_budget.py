"""BudgetService: a thin, honest wrapper — reserve passes through, never past
the limit. Atomicity itself is pinned by the store contract and
tests/test_concurrency.py; here we pin the service surface.
"""

from __future__ import annotations

from services.budget import BudgetService
from services.connectors import InMemoryStateStore

TTL = 60


async def test_reserve_charges_until_limit() -> None:
    svc = BudgetService(InMemoryStateStore())
    results = [await svc.reserve_generation("sess-1", limit=2, ttl_seconds=TTL) for _ in range(4)]
    assert results == [True, True, False, False]


async def test_sessions_do_not_share_budget() -> None:
    svc = BudgetService(InMemoryStateStore())
    assert await svc.reserve_generation("sess-1", limit=1, ttl_seconds=TTL) is True
    assert await svc.reserve_generation("sess-1", limit=1, ttl_seconds=TTL) is False
    assert await svc.reserve_generation("sess-2", limit=1, ttl_seconds=TTL) is True
