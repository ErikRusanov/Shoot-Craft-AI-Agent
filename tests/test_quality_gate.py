"""Quality gate + vision: metrics → verdict, and the profile that outlives a session.

Three rings, by prerequisite:

1. **Pure** — gate thresholds over hand-built ``FrameMetrics`` and the
   sharpness math over synthetic frames (uniform fill, generated noise,
   programmatic blur). No weights, no fixtures, never skipped.
2. **Stubbed vision** — the full ``VisionService`` path with a
   ``ScriptedFaceAnalyzer``, including the no-face uniform frame and the
   profile-reuse-across-sessions guarantee. Never skipped either.
3. **Real** — the same path through actual InsightFace on real photos; skips
   with instructions when weights/fixtures are absent (see ``tests/fixtures``).
"""

from __future__ import annotations

import io
from functools import cache
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageFilter

from protocols import DetectedFace, FaceAnalyzer
from schemas import FrameMetrics, GateReason, Verdict
from services.connectors import InMemoryStateStore
from services.quality_gate import GateThresholds, QualityGate
from services.vision import VisionService
from tests import fixtures
from tests.fakes import ScriptedFaceAnalyzer
from utils import images

THRESHOLDS = GateThresholds(
    min_side=256,
    min_face_side=128.0,
    max_secondary_face_ratio=0.25,
    min_blur_var=50.0,
    min_brightness=40.0,
    max_brightness=230.0,
)
GATE = QualityGate(THRESHOLDS)


def make_metrics(**overrides: Any) -> FrameMetrics:
    """A metrics set that passes every THRESHOLDS check unless overridden."""
    values: dict[str, Any] = {
        "face_count": 1,
        "face_area_ratio": 0.25,
        "face_side": 320.0,
        "secondary_face_ratio": 0.0,
        "blur_var": 500.0,
        "yaw": 5.0,
        "pitch": -3.0,
        "roll": 1.0,
        "brightness": 128.0,
        "width": 640,
        "height": 640,
    }
    values.update(overrides)
    return FrameMetrics(**values)


def noise_frame(side: int = 640) -> Image.Image:
    """Maximally sharp synthetic content: iid pixel noise, mean luma ~127."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def uniform_frame(side: int = 640, value: int = 128) -> Image.Image:
    return Image.new("RGB", (side, side), (value, value, value))


def grainy_soft_frame(side: int = 640, *, grain_sigma: float = 8.0) -> Image.Image:
    """A high-ISO look: soft (blurred) content under sharp sensor grain."""
    soft = np.asarray(
        noise_frame(side).filter(ImageFilter.GaussianBlur(radius=8)), dtype=np.float64
    )
    grain = np.random.default_rng(3).normal(0.0, grain_sigma, soft.shape)
    return Image.fromarray(np.clip(soft + grain, 0, 255).astype(np.uint8), mode="RGB")


def detected_face(*, side: int = 640, **overrides: Any) -> DetectedFace:
    """A clean centered face detection covering 25% of a ``side``-px frame."""
    quarter = side // 4
    values: dict[str, Any] = {
        "bbox": (quarter, quarter, side - quarter, side - quarter),
        "det_score": 0.9,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "gender": None,
        "age": 30,
        "embedding": np.full(512, 1 / np.sqrt(512), dtype=np.float32),
    }
    values.update(overrides)
    return DetectedFace(**values)


# --- ring 1: the gate is a pure function of metrics --------------------------


def test_good_metrics_pass() -> None:
    result = GATE.evaluate(make_metrics())
    assert (result.verdict, result.reason) == (Verdict.PASSED, GateReason.OK)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"face_count": 0}, GateReason.NO_FACE),
        ({"face_count": 2, "secondary_face_ratio": 0.8}, GateReason.MULTIPLE_FACES),
        ({"width": 200, "height": 640}, GateReason.LOW_RESOLUTION),
        ({"face_side": 64.0}, GateReason.FACE_TOO_SMALL),
        ({"blur_var": 10.0}, GateReason.BLURRY),
        ({"brightness": 15.0}, GateReason.POOR_LIGHTING),
        ({"brightness": 250.0}, GateReason.POOR_LIGHTING),
    ],
)
def test_each_threshold_names_its_reason(overrides: dict[str, Any], reason: GateReason) -> None:
    result = GATE.evaluate(make_metrics(**overrides))
    assert result.verdict is Verdict.BELOW_FLOOR
    assert result.reason is reason


def test_pose_is_composition_not_a_quality_problem() -> None:
    # A turned/tilted head is the user's photo, not a rendering defect — the
    # gate judges only how well the face is rendered.
    result = GATE.evaluate(make_metrics(yaw=80.0, pitch=-45.0, roll=50.0))
    assert (result.verdict, result.reason) == (Verdict.PASSED, GateReason.OK)


def test_background_bystanders_do_not_fail_the_gate() -> None:
    # Real photos have small faces in the background; only a *comparable*
    # second face is an identity ambiguity.
    result = GATE.evaluate(make_metrics(face_count=3, secondary_face_ratio=0.08))
    assert (result.verdict, result.reason) == (Verdict.PASSED, GateReason.OK)


def test_most_fundamental_failure_wins() -> None:
    # A frame with no face is also "blurry" by the numbers; the user must be
    # told about the face, not the blur.
    result = GATE.evaluate(make_metrics(face_count=0, blur_var=0.0, brightness=5.0))
    assert result.reason is GateReason.NO_FACE


# --- ring 1: sharpness math on synthetic frames ------------------------------


def test_uniform_fill_has_zero_laplacian_variance() -> None:
    assert images.laplacian_variance(images.grayscale(uniform_frame())) == 0.0


def test_programmatic_blur_collapses_laplacian_variance() -> None:
    sharp = noise_frame()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=8))
    sharp_var = images.laplacian_variance(images.grayscale(sharp))
    blurred_var = images.laplacian_variance(images.grayscale(blurred))
    assert sharp_var > THRESHOLDS.min_blur_var
    assert blurred_var < THRESHOLDS.min_blur_var
    assert blurred_var < sharp_var / 10


def test_sensor_grain_does_not_fake_sharpness() -> None:
    # Grain is pure high frequency: on the raw image the Laplacian variance of
    # a soft-but-noisy frame reads "sharp". The median prefilter is what lets
    # the gate see through it — this pins that, guarding the bad_qual.png case
    # without needing the fixture.
    frame = grainy_soft_frame()
    raw_var = images.laplacian_variance(images.grayscale(frame))
    denoised_var = images.laplacian_variance(images.grayscale(images.denoise_median(frame)))
    assert raw_var > THRESHOLDS.min_blur_var  # the lie
    assert denoised_var < THRESHOLDS.min_blur_var  # the truth


# --- ring 2: vision over a scripted analyzer (no weights, never skipped) -----


def make_vision(analyzer: FaceAnalyzer) -> VisionService:
    return VisionService(analyzer, GATE)


async def test_uniform_frame_yields_no_face_profile() -> None:
    vision = make_vision(ScriptedFaceAnalyzer([]))
    profile = await vision.build_face_profile(
        images.encode_jpeg(uniform_frame()), face_key="f-no-face", photo_ref="ref"
    )
    assert profile.gate_verdict is Verdict.BELOW_FLOOR
    assert profile.gate_reason is GateReason.NO_FACE
    assert profile.metrics.face_count == 0
    assert profile.embedding == []
    assert profile.gender is None and profile.age is None


async def test_sharp_frame_with_face_passes() -> None:
    face = detected_face()
    vision = make_vision(ScriptedFaceAnalyzer([face]))
    profile = await vision.build_face_profile(
        images.encode_jpeg(noise_frame()), face_key="f-sharp", photo_ref="ref"
    )
    assert profile.gate_verdict is Verdict.PASSED
    assert profile.gate_reason is GateReason.OK
    assert profile.metrics.face_count == 1
    assert profile.metrics.face_area_ratio == pytest.approx(0.25, abs=0.01)
    assert profile.metrics.face_side == pytest.approx(320.0)
    assert profile.embedding == pytest.approx(face.embedding.tolist())
    assert profile.age == 30


async def test_small_background_face_passes_but_comparable_one_fails() -> None:
    primary = detected_face()
    background = detected_face(bbox=(0.0, 0.0, 64.0, 64.0))  # a passer-by far behind
    comparable = detected_face(bbox=(0.0, 0.0, 320.0, 320.0))  # someone right next to them
    frame = images.encode_jpeg(noise_frame())

    vision = make_vision(ScriptedFaceAnalyzer([primary, background]))
    profile = await vision.build_face_profile(frame, face_key="f-bg", photo_ref="ref")
    assert profile.metrics.secondary_face_ratio == pytest.approx(0.04)
    assert profile.gate_verdict is Verdict.PASSED

    vision = make_vision(ScriptedFaceAnalyzer([primary, comparable]))
    profile = await vision.build_face_profile(frame, face_key="f-two", photo_ref="ref")
    assert profile.gate_reason is GateReason.MULTIPLE_FACES


async def test_programmatically_blurred_frame_fails_blurry() -> None:
    blurred = noise_frame().filter(ImageFilter.GaussianBlur(radius=8))
    vision = make_vision(ScriptedFaceAnalyzer([detected_face()]))
    profile = await vision.build_face_profile(
        images.encode_jpeg(blurred), face_key="f-blur", photo_ref="ref"
    )
    assert profile.gate_verdict is Verdict.BELOW_FLOOR
    assert profile.gate_reason is GateReason.BLURRY


async def test_grainy_soft_frame_fails_blurry() -> None:
    # PNG, not JPEG: lossy encoding would smear the very grain this case is about.
    buf = io.BytesIO()
    grainy_soft_frame().save(buf, format="PNG")
    vision = make_vision(ScriptedFaceAnalyzer([detected_face()]))
    profile = await vision.build_face_profile(buf.getvalue(), face_key="f-grain", photo_ref="ref")
    assert profile.gate_verdict is Verdict.BELOW_FLOOR
    assert profile.gate_reason is GateReason.BLURRY


async def test_face_profile_is_reused_across_sessions() -> None:
    # Session one builds the profile and stores it under face_key; session two
    # must find the identical profile without triggering a second analysis.
    analyzer = ScriptedFaceAnalyzer([detected_face()])
    vision = make_vision(analyzer)
    profile = await vision.build_face_profile(
        images.encode_jpeg(noise_frame()), face_key="face-shared", photo_ref="photos/face-shared"
    )

    store = InMemoryStateStore()
    await store.put_face(profile, ttl_seconds=3600)

    reused = await store.get_face("face-shared")
    assert reused == profile
    assert analyzer.calls == 1


# --- ring 3: the real pipeline on local-only photos + weights ----------------

# The gate config defaults (src/config.py) — what a prod ingest would apply.
REAL_THRESHOLDS = GateThresholds(
    min_side=512,
    min_face_side=128.0,
    max_secondary_face_ratio=0.25,
    min_blur_var=80.0,
    min_brightness=50.0,
    max_brightness=230.0,
)


@cache
def _real_analyzer() -> FaceAnalyzer:
    from services.connectors.insightface_embedder import InsightFaceEmbedder

    return InsightFaceEmbedder(
        model_pack=fixtures.INSIGHTFACE_MODEL, root=fixtures.INSIGHTFACE_ROOT
    )


def _real_vision() -> VisionService:
    fixtures.require_weights()
    return VisionService(_real_analyzer(), QualityGate(REAL_THRESHOLDS))


async def test_real_photo_builds_a_passing_profile() -> None:
    vision = _real_vision()
    data = fixtures.require_fixture(fixtures.FACE_A)
    profile = await vision.build_face_profile(data, face_key="fixture-a", photo_ref="ref-a")

    assert profile.gate_verdict is Verdict.PASSED
    assert profile.gate_reason is GateReason.OK
    # Background faces are allowed; the profile must anchor the primary one.
    assert profile.metrics.face_count >= 1
    assert profile.metrics.face_side >= REAL_THRESHOLDS.min_face_side
    assert len(profile.embedding) == 512
    assert profile.gender is not None
    assert profile.age is not None and 0 < profile.age < 100


async def test_real_turned_face_passes_the_gate() -> None:
    # face_b is a ~40°-turned head: composition, not a defect — must pass.
    vision = _real_vision()
    data = fixtures.require_fixture(fixtures.FACE_B)
    profile = await vision.build_face_profile(data, face_key="fixture-b", photo_ref="ref-b")

    assert profile.gate_verdict is Verdict.PASSED
    assert abs(profile.metrics.yaw) > 25  # the premise: the head really is turned


async def test_real_noisy_dim_photo_fails_the_gate() -> None:
    # The canonical "bad rendering" fixture: high-ISO grain over a soft face.
    vision = _real_vision()
    data = fixtures.require_fixture(fixtures.IMAGES_DIR / "bad_qual.png")
    profile = await vision.build_face_profile(data, face_key="fixture-bad", photo_ref="ref-bad")

    assert profile.gate_verdict is Verdict.BELOW_FLOOR
    assert profile.gate_reason is GateReason.BLURRY


async def test_real_photo_blurred_fails_the_gate() -> None:
    vision = _real_vision()
    data = fixtures.require_fixture(fixtures.FACE_A)
    blurred = images.decode_rgb(data).filter(ImageFilter.GaussianBlur(radius=4))
    profile = await vision.build_face_profile(
        images.encode_jpeg(blurred), face_key="fixture-a-blur", photo_ref="ref-a-blur"
    )

    assert profile.gate_verdict is Verdict.BELOW_FLOOR
    assert profile.gate_reason is GateReason.BLURRY
