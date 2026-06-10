"""Download the InsightFace model pack into the configured weights directory.

The runtime connector refuses to download implicitly (a prod boot must not pull
hundreds of MB on the quiet), so this script is the one sanctioned fetch path:

    make models            # honors INSIGHTFACE_MODEL / INSIGHTFACE_ROOT from .env

Weights are never committed; ``insightface_root`` is gitignored.
"""

from __future__ import annotations

from insightface.utils.storage import ensure_available

from config import get_settings


def main() -> None:
    settings = get_settings()
    dest = ensure_available("models", settings.insightface_model, root=settings.insightface_root)
    print(f"InsightFace pack '{settings.insightface_model}' ready at {dest}")


if __name__ == "__main__":
    main()
