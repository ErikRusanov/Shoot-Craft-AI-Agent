"""In-memory store and bus for tests — the real connector, re-exported.

The in-memory implementations graduated to ``services/connectors/memory_store``
(they are the configured no-Redis wiring, not just test doubles). Tests keep
importing them from ``tests.fakes`` so the fakes package remains the single
seam for port substitution.
"""

from __future__ import annotations

from services.connectors.memory_store import InMemoryEventBus, InMemoryStateStore

__all__ = ["InMemoryEventBus", "InMemoryStateStore"]
