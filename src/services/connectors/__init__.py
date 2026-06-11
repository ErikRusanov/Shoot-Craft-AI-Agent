"""Adapters implementing the ports against real (and in-memory) backends.

Only ``api/deps.py`` imports from here — services and graph nodes see the
ports in ``protocols/`` exclusively. Which connector backs a port is decided
once, at process start, from config: ``redis_url`` set → Redis store/bus, unset
→ in-memory; ``object_storage`` picks S3 or the local directory. There is no
runtime failover between backends.
"""

from __future__ import annotations

from services.connectors.fake import FakeFaceEngine, FakeImageGenerator
from services.connectors.insightface_embedder import InsightFaceEmbedder
from services.connectors.local_storage import LocalObjectStorage
from services.connectors.memory_store import InMemoryEventBus, InMemoryStateStore
from services.connectors.openrouter_classifier import OpenRouterUseCaseClassifier
from services.connectors.openrouter_client import OpenRouterClient, parse_usage
from services.connectors.openrouter_generator import (
    NoImageGeneratedError,
    OpenRouterImageGenerator,
)
from services.connectors.openrouter_slot_filler import OpenRouterSlotFiller
from services.connectors.redis_event_bus import RedisEventBus
from services.connectors.redis_store import RedisStateStore
from services.connectors.s3_storage import S3ObjectStorage
from services.connectors.throttle import ThrottledImageGenerator

__all__ = [
    "FakeFaceEngine",
    "FakeImageGenerator",
    "InMemoryEventBus",
    "InMemoryStateStore",
    "InsightFaceEmbedder",
    "LocalObjectStorage",
    "NoImageGeneratedError",
    "OpenRouterClient",
    "OpenRouterImageGenerator",
    "OpenRouterSlotFiller",
    "OpenRouterUseCaseClassifier",
    "RedisEventBus",
    "RedisStateStore",
    "S3ObjectStorage",
    "ThrottledImageGenerator",
    "parse_usage",
]
