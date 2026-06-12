"""Application configuration.

Single source of runtime config, loaded from the environment (and an optional
`.env` in dev). Field names map case-insensitively to env vars, so
`preset_source` ← `PRESET_SOURCE`, `openrouter_api_key` ← `OPENROUTER_API_KEY`.

Secrets and connection details only — no contract lives here (that is `schemas/`).
"""

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

    # --- Connectors ---
    # True → the model-shaped ports (generator, face engine, slot filler) are
    # wired to deterministic in-process fakes: the full pipeline runs with no
    # OpenRouter key, no InsightFace weights and no money spent. Store, bus and
    # object storage still follow their own settings (redis_url / object_storage),
    # so fake mode composes with real Redis for FSM/persistence testing.
    fake_connectors: bool = False

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False  # structured JSON in prod, pretty console in dev

    # --- Redis (state store + event bus) ---
    # Set → Redis connectors; unset → in-memory. Decided once at process start,
    # never failed-over at runtime: losing Redis mid-flight is an error, not a
    # silent downgrade to state that dies with the process.
    redis_url: str | None = None  # e.g. redis://localhost:6379/0
    # The face profile outlives the session by design: a session can hang on
    # need_input/approve for a while, and a style change reuses the same profile
    # without a second heavy ingest. Both stay transient and TTL-bound.
    face_ttl_seconds: int = 48 * 60 * 60  # 48h — outlives the 24h session
    session_ttl_seconds: int = 24 * 60 * 60  # 24h
    event_stream_maxlen: int = 1000  # cap per-session events:{session_key} stream

    # --- Generation (Nano Banana 2 via OpenRouter) ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    generation_model: str = "google/gemini-3.1-flash-image-preview"
    openrouter_timeout_seconds: float = 120.0
    # Transient-only (network/429/5xx) attempts; a delivered image is never
    # retried, so this can be generous without ever double-paying.
    openrouter_retry_attempts: int = 4
    # Cheap GA text model for the LLM slot filler (structured output, $0.25/M
    # in as of 2026-06); falls back to DefaultSlotFiller on any misbehavior.
    slot_filler_model: str = "google/gemini-3.1-flash-lite"
    # Cheap text model that maps the user's free-text brief onto a use_case
    # token; falls back to deterministic token-overlap on any misbehavior. Same
    # model as the slot filler by default, so the pricing table already covers it.
    classifier_model: str = "google/gemini-3.1-flash-lite"

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
    # services/quality_gate. GATE_MIN_* / GATE_MAX_* are the clean-pass levels;
    # GATE_FLOOR_* are the hard floors — between the two the verdict is SOFT
    # (usable with the user's explicit confirmation, no quality guarantee).
    # Bands calibrated on 22 labeled real photos; revisit once the generation
    # loop shows which anchors actually produce bad output.
    gate_min_side: int = 512  # min(width, height) of the frame, px — hard
    # A second face only fails the gate when comparable to the primary;
    # background passers-by sit far below this. Hard: confirmation cannot
    # resolve whose identity to anchor.
    gate_max_secondary_face_ratio: float = 0.25  # secondary bbox area / primary bbox area
    # Absolute, not a frame fraction: the recognition model aligns to 112px, so
    # what matters is pixels on the face, however the shot is composed.
    gate_min_face_side: float = 128.0  # min side of the primary face bbox, px
    gate_floor_face_side: float = 96.0
    # On the *denoised* face crop (see FrameMetrics.blur_var). Clean faces
    # measure >= 65; shadowed/retouched-but-arguably-usable ones 11..53; only
    # outright mush (2-and-below territory) is hopeless, hence the deep floor.
    gate_min_blur_var: float = 60.0
    gate_floor_blur_var: float = 8.0
    gate_min_brightness: float = 50.0  # mean luma 0..255 on the face crop
    gate_max_brightness: float = 230.0
    gate_floor_min_brightness: float = 25.0
    gate_floor_max_brightness: float = 245.0
    # Risk-only flag, never a rejection: an extreme profile is a weak identity
    # anchor, so the user confirms before budget is spent on it.
    gate_risk_max_abs_yaw: float = 60.0

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
    # `default` fallback landed in 0.3.0; 0.4.0 dropped applies_to.gender (preset
    # schema_v 3); 0.5.0 brought preset schema_v 4 (mode, style_notes, slot
    # policy) for the brief-driven writer pipeline. A prod deploy that pulls an
    # older package fails fast at startup rather than running a stale contract.
    # New schema_v 4 fields are optional, so an older 0.4.x library still loads;
    # 0.5.0 marks the library that actually uses edit mode and locked attributes.
    # Only enforced for 'package' mode.
    preset_min_library_version: str = "0.5.0"

    # --- Generation loop / budget safety ---
    # budget_limit is supplied per session by the business service; this is only a
    # hard ceiling guarding against a runaway loop.
    max_iterations: int = 8

    # --- API hardening / backpressure ---
    # Per-process cap on concurrent upstream generation calls. The semaphore
    # sits on the ImageGenerator port, so parallel sessions queue at the model
    # instead of stampeding it.
    max_concurrent_generations: int = 4
    # Hard ceiling on one graph run (one start/resume leg, not the whole
    # session): a wedged upstream cannot hold a worker slot forever. The run
    # lock's TTL derives from this, so a crashed holder frees the session.
    session_wall_clock_seconds: int = 900
    # The ingest endpoint runs CV inline (detection + embedding); bound it so a
    # pathological input cannot park the request forever.
    ingest_timeout_seconds: float = 60.0
    # Decoded size cap for an ingested photo — reject absurd payloads before
    # they reach pillow/the detector.
    max_photo_bytes: int = 20 * 1024 * 1024

    # --- Cost / pricing ---
    # Pay-as-you-go: budget_limit is real USD. Rates come from the built-in
    # PricingTable (June-2026 OpenRouter numbers); this is a JSON object that
    # overrides any of its fields at startup, so a price change ships without a
    # deploy. Shape mirrors services.pricing.PricingTable (e.g.
    # {"model_rates": {"<model>": {"input_per_mtok": "0.5", ...}}}).
    pricing_overrides_json: str | None = None
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
