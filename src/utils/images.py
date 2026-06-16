"""Image helpers — pillow + numpy only, no OpenCV.

Decode/crop/resize/encode plus the two pixel statistics the vision service
measures (Laplacian variance for sharpness, mean luma for exposure). Pure
functions over in-memory images: no I/O, no model calls, so they are trivially
testable on synthetic frames.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter, UnidentifiedImageError

Bbox = tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


def nearest_aspect_ratio(width: int, height: int, supported: Sequence[str]) -> str:
    """The supported ``"W:H"`` string closest to ``width/height``.

    Closest in log space — ``|log(actual / candidate)|`` — so a 2:1 frame is as
    far from 1:1 as a 1:2 frame is, which plain ratio differences get wrong.
    Used in edit mode to keep the generated frame on the source photo's
    composition instead of forcing the preset's generate-mode ratio.
    """
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    if not supported:
        raise ValueError("no supported aspect ratios to choose from")
    actual = width / height

    def distance(candidate: str) -> float:
        w, _, h = candidate.partition(":")
        return abs(math.log(actual / (int(w) / int(h))))

    return min(supported, key=distance)


def decode_rgb(data: bytes) -> Image.Image:
    """Decode photo bytes into an RGB pillow image.

    Raises ``ValueError`` on undecodable bytes so callers surface one error
    type for "not an image" regardless of pillow internals.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError as exc:
        raise ValueError("input bytes are not a decodable image") from exc
    return img.convert("RGB")


def encode_jpeg(img: Image.Image, *, quality: int = 90) -> bytes:
    """Encode to JPEG bytes (the format the generator and storage speak)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def crop_bbox(img: Image.Image, bbox: Bbox, *, margin: float = 0.0) -> Image.Image:
    """Crop ``bbox`` expanded by ``margin`` (fraction of bbox size), clamped to the frame.

    Detector boxes hug the face tightly; quality metrics (blur, exposure) are
    more honest with a little context around it, hence the margin.
    """
    x1, y1, x2, y2 = bbox
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    left = max(0, int(x1 - mx))
    top = max(0, int(y1 - my))
    right = min(img.width, int(x2 + mx))
    bottom = min(img.height, int(y2 + my))
    if right <= left or bottom <= top:
        raise ValueError(f"bbox {bbox} lies outside the {img.width}x{img.height} frame")
    return img.crop((left, top, right, bottom))


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Shrink so the longer side is ``max_side``; never upscales."""
    longest = max(img.width, img.height)
    if longest <= max_side:
        return img
    scale = max_side / longest
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def upscale(img: Image.Image, factor: int) -> Image.Image:
    """Upscale ``img`` by an integer ``factor`` using LANCZOS resampling.

    ``factor <= 1`` returns the image unchanged. Intended as a post-generation
    step: the generation model outputs a fixed resolution per aspect-ratio tier;
    this lifts it before storage so downstream consumers always get a larger file.
    """
    if factor <= 1:
        return img
    return img.resize((img.width * factor, img.height * factor), Image.Resampling.LANCZOS)


def grayscale(img: Image.Image) -> NDArray[np.float64]:
    """Luma plane as a float array in 0..255."""
    return np.asarray(img.convert("L"), dtype=np.float64)


def denoise_median(img: Image.Image, *, size: int = 3) -> Image.Image:
    """A small median filter — kills sensor grain, keeps true edges.

    Sharpness must be measured on this, not the raw image: high-ISO noise is
    pure high frequency, so on a noisy-but-soft photo the raw Laplacian
    variance reads "sharp" while the face itself is mush.
    """
    return img.filter(ImageFilter.MedianFilter(size))


def laplacian_variance(gray: NDArray[np.float64]) -> float:
    """Variance of the 4-neighbour Laplacian — the classic sharpness score.

    Higher = sharper. Implemented with shifted sums instead of cv2.Laplacian;
    the wrap-around ring ``np.roll`` introduces is dropped before taking the
    variance.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(lap[1:-1, 1:-1].var())


def mean_brightness(gray: NDArray[np.float64]) -> float:
    """Mean luma in 0..255 — the exposure number the gate thresholds."""
    return float(gray.mean())
