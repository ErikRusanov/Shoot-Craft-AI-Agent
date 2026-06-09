"""Ports — the Protocol interfaces every external dependency hides behind.

No implementation lives here. Services and graph nodes type against these; DI in
``api/deps.py`` is the only place a concrete connector meets a port, and tests
substitute in-memory fakes at this same seam.
"""

from __future__ import annotations

from protocols.embedder import Embedder
from protocols.event_bus import EventBus, StreamedEvent
from protocols.generator import GeneratedImage, ImageGenerator
from protocols.object_storage import ObjectStorage
from protocols.state_store import StateStore

__all__ = [
    "Embedder",
    "EventBus",
    "GeneratedImage",
    "ImageGenerator",
    "ObjectStorage",
    "StateStore",
    "StreamedEvent",
]
