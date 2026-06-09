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
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    APP_ENV=prod \
    LOG_JSON=true \
    PRESET_SOURCE=examples \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# InsightFace model weights are NOT committed; they are downloaded or mounted at
# runtime. Redis / OpenRouter / storage are configured via env (see .env.example).
CMD ["python", "-m", "main"]
