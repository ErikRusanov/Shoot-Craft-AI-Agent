"""Face-check: cosine banding against the session's thresholds, nothing hardcoded.

The bands are pinned at their exact boundaries (>= is part of the contract:
hitting the target *is* a pass), the thresholds demonstrably come from the
argument, and the embedder's "no face" signal collapses to zero similarity
instead of an error — a frame that lost the face is a failed attempt, not a crash.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from schemas import RiskLevel, Thresholds, Verdict
from services.facecheck import FaceCheckResult, FaceCheckService
from tests.fakes import ScriptedSimilarityEmbedder, axis_embedding

THRESHOLDS = Thresholds(similarity_threshold=0.8, identity_floor=0.5, K_max_retries=3)
REFERENCE = axis_embedding()


def service(*similarities: float | None) -> FaceCheckService:
    return FaceCheckService(ScriptedSimilarityEmbedder(list(similarities)))


async def check_at(
    similarity: float | None, thresholds: Thresholds = THRESHOLDS
) -> FaceCheckResult:
    return await service(similarity).check(
        reference_embedding=REFERENCE, image=b"frame", thresholds=thresholds
    )


@pytest.mark.parametrize(
    ("similarity", "verdict", "risk"),
    [
        (0.95, Verdict.PASSED, RiskLevel.LOW),
        (0.8, Verdict.PASSED, RiskLevel.LOW),  # hitting the target is a pass
        (0.79, Verdict.SOFT, RiskLevel.MEDIUM),
        (0.5, Verdict.SOFT, RiskLevel.MEDIUM),  # hitting the floor is still soft
        (0.49, Verdict.BELOW_FLOOR, RiskLevel.HIGH),
        (0.0, Verdict.BELOW_FLOOR, RiskLevel.HIGH),
    ],
)
async def test_banding(similarity: float, verdict: Verdict, risk: RiskLevel) -> None:
    result = await check_at(similarity)
    assert result.similarity == pytest.approx(similarity, abs=1e-6)
    assert result.verdict is verdict
    assert result.risk_level is risk


async def test_identical_embedding_scores_one() -> None:
    result = await check_at(1.0)
    assert result.similarity == pytest.approx(1.0)
    assert result.verdict is Verdict.PASSED


async def test_thresholds_come_from_the_argument() -> None:
    # The same similarity lands in different bands under different presets —
    # proof the bar is the session's, not a constant baked into the service.
    strict = Thresholds(similarity_threshold=0.9, identity_floor=0.7, K_max_retries=1)
    lax = Thresholds(similarity_threshold=0.6, identity_floor=0.3, K_max_retries=1)
    assert (await check_at(0.65, strict)).verdict is Verdict.BELOW_FLOOR
    assert (await check_at(0.65, THRESHOLDS)).verdict is Verdict.SOFT
    assert (await check_at(0.65, lax)).verdict is Verdict.PASSED


async def test_no_face_in_generated_frame_is_zero_similarity() -> None:
    # ScriptedSimilarityEmbedder raises the embedder's documented ValueError
    # for a scripted None — the result must be a hard fail, not an exception.
    result = await check_at(None)
    assert result.similarity == 0.0
    assert result.verdict is Verdict.BELOW_FLOOR
    assert result.risk_level is RiskLevel.HIGH


async def test_empty_reference_embedding_is_a_caller_bug() -> None:
    with pytest.raises(ValueError, match="reference embedding is empty"):
        await service(0.9).check(reference_embedding=[], image=b"frame", thresholds=THRESHOLDS)


async def test_unnormalized_vectors_still_score_correct_cosine() -> None:
    # Cosine is scale-invariant; a vector that drifted off unit norm must shift
    # nothing. Reference scaled 3x against a scripted 0.9-cosine candidate.
    scaled_reference = [v * 3.0 for v in REFERENCE]
    result = await service(0.9).check(
        reference_embedding=scaled_reference, image=b"frame", thresholds=THRESHOLDS
    )
    assert result.similarity == pytest.approx(0.9, abs=1e-6)


async def test_zero_magnitude_candidate_raises() -> None:
    class ZeroEmbedder:
        async def embed(self, image: bytes) -> NDArray[np.float32]:
            return np.zeros(8, dtype=np.float32)

    svc = FaceCheckService(ZeroEmbedder())
    with pytest.raises(ValueError, match="zero-magnitude"):
        await svc.check(reference_embedding=REFERENCE, image=b"frame", thresholds=THRESHOLDS)


async def test_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dimensions differ"):
        await service(0.9).check(
            reference_embedding=[1.0, 0.0], image=b"frame", thresholds=THRESHOLDS
        )
