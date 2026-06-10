# Public base image for photocore. Ships the core + the MIT demo presets ONLY —
# no private library. Runs out of the box on PRESET_SOURCE=examples.
#
# The private image lives in the presets repo and builds FROM this image, adding
# the real library as a versioned wheel (see ../presets/Dockerfile). The core
# never depends on that package; it is layered on top here, not built in.

# ---- builder: resolve deps into a venv against the lock ----
FROM python:3.14-slim AS builder

# uv as the package manager (matches local dev).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Deps first, against the lock, for layer caching. This is an application
# (tool.uv.package = false): uv builds no project, so only the manifest is
# needed here — source is copied afterwards and never invalidates this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application source (flat src/, on PYTHONPATH) + the demo preset set.
COPY src/ ./src/
COPY presets/ ./presets/

# ---- runtime: slim image with only what's needed to run ----
FROM python:3.14-slim AS runtime

# Native libs for onnxruntime / insightface / opencv (the CV face-check path).
# onnxruntime is the CPU-only wheel (no CUDA libs pulled in) — CPU is enough
# for the face-check path by design.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

# Non-root: the app needs to write only the weights dir and the local-storage
# fallback; everything else stays root-owned and read-only to the process.
RUN groupadd --system app && useradd --system --gid app --no-create-home app \
    && mkdir -p /app/.models /app/.storage \
    && chown app:app /app/.models /app/.storage

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    APP_ENV=prod \
    LOG_JSON=true \
    PRESET_SOURCE=examples \
    INSIGHTFACE_ROOT=/app/.models \
    LOCAL_STORAGE_PATH=/app/.storage \
    HOST=0.0.0.0 \
    PORT=8000

USER app
EXPOSE 8000

# Liveness only (readiness involves Redis and belongs to the orchestrator).
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz', timeout=2)"]

# InsightFace model weights are NOT committed and NOT baked into the image:
# mount them into /app/.models (or run scripts/download_models.py against the
# volume once). Redis / OpenRouter / storage are configured via env
# (see .env.example).
CMD ["python", "-m", "main"]
