"""Contract: :class:`~protocols.embedder.Embedder`.

The embedder's input domain differs by implementation — arbitrary bytes for the
fake, a real face photo for InsightFace — so each parameter supplies its own
sample images alongside the implementation. The assertions are about the *vector
relationship*, which holds regardless: identical input embeds identically, and
two distinct identities embed apart.

The InsightFace case needs local-only prerequisites (weights + photo fixtures,
see ``tests/fixtures``) and skips with instructions when they are absent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest

from protocols import Embedder
from tests import fixtures
from tests.fakes import DeterministicEmbedder


@dataclass
class EmbedCase:
    """An embedder plus two byte blobs standing for two distinct faces."""

    embedder: Embedder
    image_a: bytes
    image_b: bytes


def _fake_case() -> EmbedCase:
    return EmbedCase(
        embedder=DeterministicEmbedder(),
        image_a=b"identity-A-photo",
        image_b=b"identity-B-photo",
    )


@cache
def _insightface_embedder() -> Embedder:
    # Cached: loading the ONNX pack costs ~a second and the instance is
    # stateless across calls — one is enough for the whole module.
    from services.connectors.insightface_embedder import InsightFaceEmbedder

    return InsightFaceEmbedder(
        model_pack=fixtures.INSIGHTFACE_MODEL, root=fixtures.INSIGHTFACE_ROOT
    )


def _insightface_case() -> EmbedCase:
    fixtures.require_weights()
    image_a = fixtures.require_fixture(fixtures.FACE_A)
    image_b = fixtures.require_fixture(fixtures.FACE_B)
    return EmbedCase(embedder=_insightface_embedder(), image_a=image_a, image_b=image_b)


CASE_FACTORIES = [
    pytest.param(_fake_case, id="deterministic"),
    pytest.param(_insightface_case, id="insightface"),
]


@pytest.fixture(params=CASE_FACTORIES)
def case(request: pytest.FixtureRequest) -> EmbedCase:
    factory: Callable[[], EmbedCase] = request.param
    return factory()


async def test_is_an_embedder(case: EmbedCase) -> None:
    assert isinstance(case.embedder, Embedder)


async def test_returns_a_finite_1d_vector(case: EmbedCase) -> None:
    vec = await case.embedder.embed(case.image_a)
    assert vec.ndim == 1 and vec.size > 0
    assert np.all(np.isfinite(vec))


async def test_same_input_embeds_identically(case: EmbedCase) -> None:
    first = await case.embedder.embed(case.image_a)
    second = await case.embedder.embed(case.image_a)
    assert np.array_equal(first, second)


async def test_distinct_identities_embed_apart(case: EmbedCase) -> None:
    a = await case.embedder.embed(case.image_a)
    b = await case.embedder.embed(case.image_b)
    assert a.shape == b.shape
    # Same person ≈ 1.0; different people sit well below it.
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    self_cosine = float(a @ a / (np.linalg.norm(a) ** 2))
    assert cosine < self_cosine
