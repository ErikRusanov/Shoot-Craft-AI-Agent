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
    # Cosine similarity threshold for "same identity"; tuned against the model pack.
    face_match_threshold: float = 0.35

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

    # --- Generation loop / budget safety ---
    # budget_limit is supplied per session by the business service; this is only a
    # hard ceiling guarding against a runaway loop.
    max_iterations: int = 8

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
