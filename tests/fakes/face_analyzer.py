"""Scripted face analyzer fake — returns a fixed detection result.

Lets vision/gate tests dictate exactly what was "detected" (no faces, one face
with chosen pose/size, several faces) without weights or real inference. The
call counter exists for the profile-reuse assertion: a profile served from the
store must not trigger a second analysis.
"""

from __future__ import annotations

from protocols import DetectedFace


class ScriptedFaceAnalyzer:
    """Always answers ``analyze`` with the faces it was constructed with."""

    def __init__(self, faces: list[DetectedFace]) -> None:
        self._faces = faces
        self.calls = 0

    async def analyze(self, image: bytes) -> list[DetectedFace]:
        self.calls += 1
        return list(self._faces)
