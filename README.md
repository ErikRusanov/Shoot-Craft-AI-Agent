# Shoot Craft AI Agent

**AI photoshoot core** — takes a user's reference photo, runs a face-preserving generation loop, and delivers studio-quality shots with their real identity. No prompting from the user.

![Python 3.14](https://img.shields.io/badge/python-3.14-blue) ![License AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)

This is the autonomous worker. It has no UI and never talks to a human directly — a **business service** drives it over HTTP and receives results via SSE. See [docs/business-flow.md](docs/business-flow.md) for the end-to-end flow.

---

## Architecture

```
┌─────────────────────────────────────┐
│  FastAPI  ·  SSE stream             │  src/api/
├─────────────────────────────────────┤
│  LangGraph FSM  ·  8-state machine  │  src/graph/
├─────────────────────────────────────┤
│  Domain services  (no I/O)          │  src/services/
├─────────────────────────────────────┤
│  Protocol ports  (interfaces)       │  src/protocols/
├───────────┬─────────┬───────────────┤
│ Generator │ Storage │ Face CV       │  src/services/connectors/
│ OpenRouter│ S3/local│ InsightFace   │
└───────────┴─────────┴───────────────┘
         Redis  (state + events)
```

Full diagram → [docs/architecture.md](docs/architecture.md)  
State machine → [docs/fsm.md](docs/fsm.md)

---

## Quick start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker (for Redis; or bring your own)

### Dev — fake connectors, no API keys or model weights needed

```bash
make setup          # install uv, sync deps, create .env, wire git hooks
make infra          # start Redis in Docker
make run            # start the API server on :8000
```

Or run the full stack (app + Redis) in containers:

```bash
make up
```

### Production / real connectors

Set the following in `.env` (see `.env.example` for all options):

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter key for the generation model |
| `REDIS_URL` | Redis connection string |
| `OBJECT_STORAGE` | `s3` or `local` (default) |
| `PRESET_SOURCE` | `package` (prod), `path` (dev), `examples` (default) |

```bash
make models         # download InsightFace weights to ./.models
make run
```

To use the private preset library locally:

```bash
# Option A — read the sibling checkout directly (edits are live):
#   PRESET_SOURCE=path  PRESET_LIBRARY_PATH=../presets/src/library  in .env

# Option B — install as a package (mirrors prod):
make presets-dev    # editable-install ../presets; set PRESET_SOURCE=package
```

---

## API

Seven endpoints; every mutation is idempotent via `idem_key`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/faces/{face_key}` | Ingest photo, run quality gate |
| `POST` | `/v1/sessions/{session_key}` | Start session |
| `POST` | `/v1/sessions/{session_key}/input` | Submit free-form scene answer |
| `POST` | `/v1/sessions/{session_key}/approve` | Approve or reject cost estimate |
| `POST` | `/v1/sessions/{session_key}/cancel` | Cancel session |
| `GET` | `/v1/sessions/{session_key}` | Read state snapshot |
| `GET` | `/v1/sessions/{session_key}/events` | SSE event stream |

Full reference with request/response shapes and a sequence diagram → [docs/api.md](docs/api.md)

---

## Configuration

All settings live in `.env` (template: `.env.example`). Key groups:

- **Redis** — `REDIS_URL`, `FACE_TTL_SECONDS`, `SESSION_TTL_SECONDS`
- **Generation** — `OPENROUTER_API_KEY`, `GENERATION_MODEL`, `MAX_ITERATIONS`
- **Face CV** — `INSIGHTFACE_MODEL`, `INSIGHTFACE_ROOT`, `FACE_MATCH_THRESHOLD`
- **Storage** — `OBJECT_STORAGE`, `S3_BUCKET`, `LOCAL_STORAGE_PATH`
- **Limits** — `MAX_CONCURRENT_GENERATIONS`, `SESSION_WALL_CLOCK_SECONDS`, `MAX_PHOTO_BYTES`
- **Fake connectors** — `FAKE_CONNECTORS=true` (no external deps, for dev/CI)

---

## Preset library

The preset library is the product moat and lives in a **separate private repo** (`photocore-presets`). This public core ships only:

- `src/schemas/presets.py` — the canonical `Preset` schema
- `presets/examples/` — 2–3 MIT-licensed demo presets for tests and local dev

The private library is loaded at runtime via `PRESET_SOURCE`. A production image layers it on via a second Dockerfile in the presets repo; see the Docker section in the original README for the two-image build pattern.

---

## Development

```bash
make help           # list all targets
make lint           # ruff check + format check
make fmt            # auto-format
make type           # mypy strict (src + tests)
make test           # pytest
make test ARGS=tests/test_x.py::test_name   # single test
make load           # parallel load test against a running server
```

Commit prefix convention: `(feat):`, `(fix):`, `(chore):`, `(refactor):`, `(docs):`, `(test):`, `(perf):`.

---

## License

[AGPL-3.0](LICENSE)
