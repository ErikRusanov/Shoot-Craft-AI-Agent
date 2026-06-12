"""Ports — the Protocol interfaces every external dependency hides behind.

No implementation lives here. Services and graph nodes type against these; DI in
``api/deps.py`` is the only place a concrete connector meets a port, and tests
substitute in-memory fakes at this same seam.
"""

from __future__ import annotations

from protocols.brief_parser import BriefParser, ParseResult
from protocols.budget import BudgetMeter, BudgetReservation
from protocols.embedder import Embedder
from protocols.event_bus import EventBus, StreamedEvent
from protocols.face_analyzer import DetectedFace, FaceAnalyzer
from protocols.generator import GeneratedImage, GenerationRefusedError, ImageGenerator
from protocols.inventory import InventoryExtractor, InventoryResult
from protocols.object_storage import ObjectStorage
from protocols.planner import PlanResult, StepPlanner
from protocols.prompt_writer import (
    PromptWriter,
    WriteRequest,
    WriteResult,
    WriterFeedback,
)
from protocols.slot_filler import SlotFill, SlotFiller
from protocols.state_store import StateStore

__all__ = [
    "BriefParser",
    "BudgetMeter",
    "BudgetReservation",
    "DetectedFace",
    "Embedder",
    "EventBus",
    "FaceAnalyzer",
    "GeneratedImage",
    "GenerationRefusedError",
    "ImageGenerator",
    "InventoryExtractor",
    "InventoryResult",
    "ObjectStorage",
    "ParseResult",
    "PlanResult",
    "PromptWriter",
    "SlotFill",
    "SlotFiller",
    "StateStore",
    "StepPlanner",
    "StreamedEvent",
    "WriteRequest",
    "WriteResult",
    "WriterFeedback",
]
