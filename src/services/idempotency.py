"""Idempotency — run a mutation once per ``idem_key``, replay it ever after.

Every mutating endpoint carries an ``idem_key``; a retried request must observe
the original outcome, not redo the work (and certainly not double-charge the
budget). The store provides set-if-absent storage; this service wraps it into
the run-once discipline, including the race where two concurrent calls with
the same key both find the cache empty — the loser of the ``idem_put`` race
discards its own result and returns the winner's.

Results are opaque ``bytes`` (callers serialize/deserialize their own payloads)
and live under the same TTL regime as the session.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from protocols import StateStore


class IdempotencyService:
    """Execute-once semantics for mutations keyed by ``idem_key``."""

    def __init__(self, store: StateStore) -> None:
        self._store = store

    async def run_once(
        self,
        idem_key: str,
        *,
        ttl_seconds: int,
        op: Callable[[], Awaitable[bytes]],
    ) -> tuple[bytes, bool]:
        """Run ``op`` unless a result for ``idem_key`` already exists.

        Returns ``(result, replayed)``: ``replayed`` is ``True`` when the
        result came from the store instead of this call's ``op``.
        """
        cached = await self._store.idem_get(idem_key)
        if cached is not None:
            return cached, True

        result = await op()
        stored = await self._store.idem_put(idem_key, result, ttl_seconds=ttl_seconds)
        if stored:
            return result, False

        # Lost the put race: a concurrent call finished first. Its result is
        # the canonical one — ours is discarded, both callers see one outcome.
        winner = await self._store.idem_get(idem_key)
        if winner is None:
            # Stored result expired between our put and get; vanishingly rare
            # (TTLs are hours) and not silently recoverable — the two results
            # may differ and we can no longer tell which one was observed.
            raise RuntimeError(f"idempotency record vanished for key {idem_key!r}")
        return winner, True
