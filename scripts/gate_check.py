"""Manual gate check: run any photos through vision + quality gate, print verdicts.

The hands-on tuning tool for the ingest front while it is not yet wired into
the API (that happens with the graph in step 8): drop in your own photos, see
exactly which metric passes or fails them under the *current config* thresholds.

    make models                                   # once: weights
    PYTHONPATH=src uv run python scripts/gate_check.py ~/Photos/me*.jpg

No PII leaves the machine and nothing is stored — profiles are built and printed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from config import get_settings
from services.connectors.insightface_embedder import InsightFaceEmbedder
from services.quality_gate import GateThresholds, QualityGate
from services.vision import VisionService


def gate_from_settings() -> tuple[QualityGate, GateThresholds]:
    s = get_settings()
    t = GateThresholds(
        min_side=s.gate_min_side,
        min_face_side=s.gate_min_face_side,
        max_secondary_face_ratio=s.gate_max_secondary_face_ratio,
        min_blur_var=s.gate_min_blur_var,
        min_brightness=s.gate_min_brightness,
        max_brightness=s.gate_max_brightness,
    )
    return QualityGate(t), t


async def main(paths: list[Path]) -> None:
    settings = get_settings()
    gate, t = gate_from_settings()
    vision = VisionService(
        InsightFaceEmbedder(
            model_pack=settings.insightface_model,
            root=settings.insightface_root,
            det_size=settings.insightface_det_size,
        ),
        gate,
    )

    print(
        f"thresholds: frame>={t.min_side}px face>={t.min_face_side:.0f}px "
        f"secondary<={t.max_secondary_face_ratio} blur>={t.min_blur_var} "
        f"brightness {t.min_brightness:.0f}..{t.max_brightness:.0f}\n"
    )
    for path in paths:
        p = await vision.build_face_profile(
            path.read_bytes(), face_key=path.stem, photo_ref=str(path)
        )
        m = p.metrics
        mark = "PASS" if p.gate_reason.value == "ok" else f"FAIL: {p.gate_reason.value}"
        print(
            f"{path.name:30s} {mark:22s} "
            f"faces={m.face_count} frame={m.width}x{m.height} face_side={m.face_side:.0f} "
            f"secondary={m.secondary_face_ratio:.2f} blur={m.blur_var:.0f} "
            f"bright={m.brightness:.0f} yaw={m.yaw:+.0f} "
            f"gender={p.gender.value if p.gender else '?'} age={p.age or '?'}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: PYTHONPATH=src uv run python scripts/gate_check.py <photo> [photo ...]")
    asyncio.run(main([Path(a) for a in sys.argv[1:]]))
