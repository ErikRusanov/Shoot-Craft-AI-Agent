"""Helpers for tests that need real photos and/or InsightFace weights.

Both are local-only prerequisites (see ``images/README.md``): photos of real
people are never committed, weights are never committed. The ``require_*``
helpers skip the calling test with an actionable message when either is
missing, so CI without them stays green and a developer knows exactly what to
provide.

The weights check duplicates the connector's three-line path convention on
purpose — importing the connector's helper would be free (it imports
insightface lazily), but the tests' skip decision must not depend on the very
code under test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

IMAGES_DIR = Path(__file__).parent / "images"
FACE_A = IMAGES_DIR / "face_a.jpg"
FACE_B = IMAGES_DIR / "face_b.jpg"

# Mirror the config defaults; honor the same env vars Settings reads.
INSIGHTFACE_ROOT = os.environ.get("INSIGHTFACE_ROOT", "./.models")
INSIGHTFACE_MODEL = os.environ.get("INSIGHTFACE_MODEL", "buffalo_l")


def weights_available() -> bool:
    pack_dir = Path(INSIGHTFACE_ROOT).expanduser() / "models" / INSIGHTFACE_MODEL
    return any(pack_dir.glob("*.onnx"))


def require_weights() -> None:
    if not weights_available():
        pytest.skip(
            f"InsightFace weights ('{INSIGHTFACE_MODEL}') not found under "
            f"{INSIGHTFACE_ROOT} — run `make models` to enable these tests"
        )


def require_fixture(path: Path) -> bytes:
    if not path.is_file():
        pytest.skip(
            f"fixture photo {path.relative_to(IMAGES_DIR.parent)} is missing — "
            "see tests/fixtures/images/README.md for what to provide"
        )
    return path.read_bytes()
