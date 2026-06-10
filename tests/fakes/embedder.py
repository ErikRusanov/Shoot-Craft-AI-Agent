"""Deterministic embedder fake — same bytes in, same vector out.

Derives a unit vector from a hash of the image bytes, so identical bytes embed to
an identical vector (cosine 1.0) and different bytes to a near-orthogonal one.
No face detection, no model weights — just a stable stand-in for InsightFace at
the :class:`~protocols.embedder.Embedder` port.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray


class DeterministicEmbedder:
    """A pure hash → unit-vector embedder. ``dim`` matches buffalo_l (512)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    async def embed(self, image: bytes) -> NDArray[np.float32]:
        seed = int.from_bytes(hashlib.sha256(image).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec


def axis_embedding(dim: int = 8) -> list[float]:
    """The reference unit vector :class:`ScriptedSimilarityEmbedder` scores against."""
    vec = [0.0] * dim
    vec[0] = 1.0
    return vec


class ScriptedSimilarityEmbedder:
    """Plays back a queue of cosine similarities against :func:`axis_embedding`.

    Each ``embed`` call pops the next scripted value ``c`` and returns the unit
    vector ``(c, sqrt(1 - c²), 0, …)`` — whose cosine with the axis reference is
    exactly ``c``. Loop tests script the per-attempt similarity sequence this
    way without faking the face-check service itself. A scripted ``None`` raises
    the embedder's "no face detected" ``ValueError`` instead.
    """

    def __init__(self, similarities: list[float | None], dim: int = 8) -> None:
        self._queue = list(similarities)
        self.dim = dim

    async def embed(self, image: bytes) -> NDArray[np.float32]:
        if not self._queue:
            raise AssertionError("ScriptedSimilarityEmbedder: more embed calls than scripted")
        sim = self._queue.pop(0)
        if sim is None:
            raise ValueError("no face detected — cannot embed")
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = sim
        vec[1] = np.sqrt(max(0.0, 1.0 - sim * sim))
        return vec
