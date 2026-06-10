"""Application configuration.

Single source of runtime config, loaded from the environment (and an optional
`.env` in dev). Field names map case-insensitively to env vars, so
`preset_source` ← `PRESET_SOURCE`, `openrouter_api_key` ← `OPENROUTER_API_KEY`.

Secrets and connection details only — no contract lives here (that is `schemas/`).
"""

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App / server ---
    app_env: Literal["dev", "prod"] = "dev"
    host: str = "0.0.0.0"  # container-internal bind; the edge handles exposure
    port: int = 8000
    reload: bool = False  # uvicorn autoreload, dev only

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False  # structured JSON in prod, pretty console in dev

    # --- Redis (state store + event bus) ---
    # Set → Redis connectors; unset → in-memory. Decided once at process start,
    # never failed-over at runtime: losing Redis mid-flight is an error, not a
    # silent downgrade to state that dies with the process.
    redis_url: str | None = None  # e.g. redis://localhost:6379/0
    # Biometrics are transient — face profiles expire fast; sessions live longer.
    face_ttl_seconds: int = 60 * 60  # 1h
    session_ttl_seconds: int = 24 * 60 * 60  # 24h
    event_stream_maxlen: int = 1000  # cap per-session events:{session_key} stream

    # --- Generation (Nano Banana 2 via OpenRouter) ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    generation_model: str = "google/gemini-3.1-flash-image-preview"
    openrouter_timeout_seconds: float = 120.0

    # --- Face check (CV, not LLM) ---
    insightface_model: str = "buffalo_l"
    # Weights live outside the repo and the image; `make models` downloads the
    # pack here. The connector refuses to silently download at startup.
    insightface_root: str = "./.models"
    insightface_det_size: int = 640  # detector input side, px
    # Cosine similarity threshold for "same identity"; tuned against the model pack.
    face_match_threshold: float = 0.35

    # --- Input-photo quality gate ---
    # Thresholds on FrameMetrics; deps assembles them into a GateThresholds for
    # services/quality_gate. Defaults are starting points, expected to be tuned.
    gate_min_side: int = 512  # min(width, height) of the frame, px
    # Absolute, not a frame fraction: the recognition model aligns to 112px, so
    # what matters is pixels on the face, however the shot is composed.
    gate_min_face_side: float = 128.0  # min side of the primary face bbox, px
    # A second face only fails the gate when comparable to the primary;
    # background passers-by sit far below this.
    gate_max_secondary_face_ratio: float = 0.25  # secondary bbox area / primary bbox area
    # On the *denoised* face crop (see FrameMetrics.blur_var). Calibrated on
    # 22 labeled real photos: usable faces measure >= 65, soft/noisy/heavily
    # retouched ones <= 53. The bands sit close — revisit once the generation
    # loop shows which anchors actually produce bad output.
    gate_min_blur_var: float = 60.0
    gate_min_brightness: float = 50.0  # mean luma 0..255 on the face crop
    gate_max_brightness: float = 230.0

    # --- Object storage ---
    object_storage: Literal["s3", "local"] = "local"
    local_storage_path: str = "./.storage"
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    s3_bucket: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    # --- Presets ---
    # package (prod, importlib.resources), path (dev/custom dir), or the in-repo
    # examples/ fallback when nothing is configured.
    preset_source: Literal["package", "path", "examples"] = "examples"
    # Distribution to read in 'package' mode. Only a name as a string lives here —
    # the core never depends on this package; the private image installs it.
    preset_package: str = "photocore_presets"
    preset_library_path: str | None = None
    # Runtime expectation on the curated library in 'package' mode. The reserved
    # `default` fallback preset (and its convention) landed in 0.3.0; a prod
    # deploy that pulls an older package fails fast at startup rather than
    # silently losing the fallback. Only enforced for 'package' mode.
    preset_min_library_version: str = "0.3.0"

    # --- Generation loop / budget safety ---
    # budget_limit is supplied per session by the business service; this is only a
    # hard ceiling guarding against a runaway loop.
    max_iterations: int = 8

    # --- Cost estimation ---
    # Price of one paid generation in abstract units; the business service maps
    # units to real money. Used only for the plan's CostEstimate. Decimal, not
    # float — it is price-like, the arithmetic must stay exact.
    generation_unit_price: Decimal = Decimal("1.0")
    # Fallback expected paid generations when a preset ships no convergence stats.
    default_expected_generations: int = 3

    @model_validator(mode="after")
    def _check_path_preset(self) -> Settings:
        if self.preset_source == "path" and not self.preset_library_path:
            raise ValueError("preset_library_path is required when preset_source='path'")
        return self

    @model_validator(mode="after")
    def _check_s3(self) -> Settings:
        if self.object_storage == "s3" and not self.s3_bucket:
            raise ValueError("s3_bucket is required when object_storage='s3'")
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — load env once per process."""
    return Settings()
