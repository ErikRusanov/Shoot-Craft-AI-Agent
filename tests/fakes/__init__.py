"""Test doubles for the ports — in-memory, deterministic, no external services.

These let the pipeline run end to end without Redis, OpenRouter or InsightFace.
The contract tests in ``tests/contract`` pin them to their ports; later the same
tests run against the real connectors.
"""

from __future__ import annotations

from tests.fakes.embedder import DeterministicEmbedder
from tests.fakes.generator import FixedImageGenerator, GenerateCall
from tests.fakes.slot_filler import FillCall, FixedSlotFiller
from tests.fakes.storage import InMemoryObjectStorage
from tests.fakes.store import InMemoryEventBus, InMemoryStateStore

__all__ = [
    "DeterministicEmbedder",
    "FillCall",
    "FixedImageGenerator",
    "FixedSlotFiller",
    "GenerateCall",
    "InMemoryEventBus",
    "InMemoryObjectStorage",
    "InMemoryStateStore",
]
