"""Port: face embedder — turns one face image into an identity vector.

Face-check is CV, not an LLM: identity is decided by cosine distance between
these vectors, so the *only* thing the core needs from the embedder is the
vector. Detection quality (bbox, pose, blur → :class:`~schemas.state.FrameMetrics`)
is a separate concern measured by the vision service; this port stays a single
pure function so a test can swap InsightFace for a deterministic stand-in.

The vector it returns is biometric data — it lives only on
:class:`~schemas.state.FaceProfile` and is **never** logged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Embedder(Protocol):
    """Embed a single detected face into a fixed-length identity vector."""

    async def embed(self, image: bytes) -> NDArray[np.float32]:
        """Return the L2-normalized embedding for the face in ``image``.

        ``image`` is the encoded photo bytes (the same bytes that go to object
        storage). Async because the real implementation offloads ONNX inference
        to a thread — every concrete embedder keeps the I/O path non-blocking.
        """
        ...
