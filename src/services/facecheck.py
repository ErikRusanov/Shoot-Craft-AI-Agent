"""Face-check — is the generated frame still the same person?

CV, not an LLM: identity is the cosine similarity between the reference
embedding (from :class:`~schemas.state.FaceProfile`, computed at ingest) and the
embedding of the generated frame, measured through the
:class:`~protocols.embedder.Embedder` port. The thresholds arrive from the
session (frozen there from the matched preset), **never** hardcoded here — a
portrait preset and a stylized one are allowed different bars.

Same three bands as the input gate, same meaning shifted to output:

- ``PASSED`` — at or above ``similarity_threshold``: deliverable, stop retrying.
- ``SOFT`` — between ``identity_floor`` and the target: shippable as keep-best
  when retries run out, but worth another attempt while budget allows.
- ``BELOW_FLOOR`` — under ``identity_floor``: never delivered, no exceptions.

A generated frame with no detectable face reads as similarity 0.0 — by
definition not the same person — rather than as an error: the model painting
the face away is a normal failed attempt the loop should retry, not a crash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from protocols import Embedder
from schemas import RiskLevel, Thresholds, Verdict

# One similarity number drives both signals, so for face-check the risk band is
# the verdict band's shadow. Risk diverging from verdict (pass-but-risky) comes
# from signals face-check does not measure (e.g. pose), upstream of here.
_RISK_BY_VERDICT: dict[Verdict, RiskLevel] = {
    Verdict.PASSED: RiskLevel.LOW,
    Verdict.SOFT: RiskLevel.MEDIUM,
    Verdict.BELOW_FLOOR: RiskLevel.HIGH,
}


@dataclass(frozen=True, slots=True)
class FaceCheckResult:
    """Measured identity similarity of one generated frame, banded."""

    similarity: float
    verdict: Verdict
    risk_level: RiskLevel


class FaceCheckService:
    """Scores a generated frame against the session's reference identity."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def check(
        self,
        *,
        reference_embedding: Sequence[float],
        image: bytes,
        thresholds: Thresholds,
    ) -> FaceCheckResult:
        """Embed ``image``, compare to the reference, band against ``thresholds``.

        ``reference_embedding`` is ``FaceProfile.embedding``; an empty one means
        the profile never passed ingest and running the loop on it is a caller
        bug, so it raises rather than scoring.
        """
        if not reference_embedding:
            raise ValueError("reference embedding is empty — the face profile has no identity")

        try:
            candidate = await self._embedder.embed(image)
        except ValueError:
            # The embedder's "no face detected" signal: the generated frame lost
            # the face entirely. Zero similarity, hard fail — retryable.
            return self._banded(0.0, thresholds)

        reference = np.asarray(reference_embedding, dtype=np.float32)
        return self._banded(_cosine(reference, candidate), thresholds)

    @staticmethod
    def _banded(similarity: float, thresholds: Thresholds) -> FaceCheckResult:
        if similarity >= thresholds.similarity_threshold:
            verdict = Verdict.PASSED
        elif similarity >= thresholds.identity_floor:
            verdict = Verdict.SOFT
        else:
            verdict = Verdict.BELOW_FLOOR
        return FaceCheckResult(
            similarity=similarity, verdict=verdict, risk_level=_RISK_BY_VERDICT[verdict]
        )


def _cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity, clipped to [-1, 1].

    Embeddings are L2-normalized by the port contract, but the division is kept
    anyway — a slightly off-norm vector should shift the score, not corrupt it.
    """
    if a.shape != b.shape:
        raise ValueError(f"embedding dimensions differ: {a.shape} vs {b.shape}")
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        raise ValueError("cannot compare a zero-magnitude embedding")
    return float(np.clip(float(np.dot(a, b)) / norm, -1.0, 1.0))
