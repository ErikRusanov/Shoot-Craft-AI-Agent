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
