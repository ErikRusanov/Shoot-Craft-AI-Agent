# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The entire repository — code, docs, comments, this file — is written in English.

## What this is

`photocore` is the core agent for a **single** AI photoshoot session. Given a
user's photos it generates photorealistic images (avatars, resume/document
photos) that preserve facial identity, in as few iterations as possible. It is
an autonomous worker driven by a trusted business service; it has no UI and
never talks to a human directly.

## Core boundary (hard rule)

The core does **not** know about or implement: users, authentication, money,
solvency, limits, anti-fraud, moderation, human chat history, or a relational
DB. All of that is the business service. If an endpoint is called, the user is
already valid and paid — the core does not re-check this. If a task pulls toward
those concerns, **stop** and flag it rather than implementing it here.

## Stack & version rule

Python 3.14, package manager **uv**. This is an application, not a library
(`tool.uv.package = false`): no build backend, source lives flat under `src/`
(`src/api`, `src/services`, …) and is on the path via `pythonpath`/`mypy_path`.

**Version rule (mandatory):** before adding or upgrading *any* dependency, look
up its current latest stable version on the web and pin that. Do not trust
versions from training data — they are stale. Verify exact package names on the
web too (e.g. the langgraph Redis checkpointer is `langgraph-checkpoint-redis`).

## Commands

Always go through the `Makefile`:

```bash
make help                          # list targets
make sync                          # install/sync env from the lock
make lint                          # lint + format check
make fmt                           # auto-format
make type                          # types (strict; src + tests)
make test                          # all tests
make test ARGS=tests/test_x.py::test_name   # a single test
make run                           # run the worker (entry lands later)
```

Adding dependencies is the one thing not wrapped: use `uv add <pkg>` /
`uv add --dev <pkg>` directly (see the version rule above).

## Architecture

Layered; dependencies point inward toward `protocols`/`schemas`:

- `schemas/` — all pydantic models and the **only** home of the contract: API
  in/out (`contract.py`), internal state (`state.py`: FaceProfile, SessionState,
  Iteration, EditStep), the brief reading (`brief.py`: BriefAnalysis, Change),
  SSE events (`events.py`), presets (`presets.py`), enums (`enums.py`). Every
  model carries a `schema_v` field for versioning.
- `protocols/` — ports (Protocol interfaces), no implementation: `Embedder`,
  `ImageGenerator`, `StateStore`, `ObjectStorage`, `EventBus`, `BriefParser`,
  `StepPlanner`, `PromptWriter`, `SlotFiller`.
- `services/` — domain logic. Depends **only** on `protocols` and `schemas`,
  never on concrete connectors: vision, quality_gate, preset_matcher,
  brief_parser, planner, prompt_writer, prompt_builder, slot_filler, facecheck,
  generation_loop, estimator, budget, idempotency, telemetry. The brief-driven
  flow is **parse → resolve constraints → plan steps → chained generation**: an
  LLM stage with a deterministic no-LLM fallback at each step (`brief_parser`,
  `planner`, `prompt_writer`), so the pipeline never fails because an LLM did.
- `services/connectors/` — adapters implementing the ports: `redis_store` +
  `memory_store` (fallback), `redis_event_bus`, `openrouter`,
  `insightface_embedder`, `s3_storage` + `local_storage` (dev fallback).
- `graph/` — LangGraph FSM orchestration. Nodes (`nodes.py`) only call services
  and hold no logic; `builder.py` assembles the graph; `state.py` is the graph
  state on top of `schemas/state`.
- `api/` — FastAPI: `app.py` (factory), `routes.py` (contract), `sse.py` (tails
  the event bus, reconnect via `Last-Event-ID`), `deps.py` (DI — the **only**
  place services meet concrete connectors).
- `utils/` — logging (structlog), images (pillow), retry (tenacity), lua (Redis
  scripts).

DI flow: `api/deps.py` binds connectors behind ports → graph nodes call services
→ services use only ports. Tests substitute at the port level.

## Domain facts

- **Generation**: Nano Banana 2 = `google/gemini-3.1-flash-image-preview` via
  OpenRouter (`chat/completions`, `modalities: ["image","text"]`, image as
  base64). This is **reference-conditioned edit**, not text2img — the model has
  no denoise/strength.
- **Face-check** uses CV (embedding + cosine), **not** an LLM. The embedder sits
  behind a port; default is InsightFace (onnxruntime, CPU is enough). Model
  weights are **not** committed.
- **State** lives in Redis keyed by `session_key`/`face_key`, no relational DB,
  in-memory fallback. Biometrics are transient, TTL-bound.
- **Streaming** via Redis Stream `events:{session_key}`; SSE tails it.
- **Budget** (`budget_limit`) is the number of paid generations; atomic Lua
  increment. All mutations are **idempotent** by `idem_key`. Results are
  **keep-best**.
- **Prompt adaptation (writer-driven).** An LLM **prompt writer**
  (`services/prompt_writer.py`, behind the `PromptWriter` port) composes the
  scene/edit **body** per situation — mode (edit/generate), the step's
  instruction, the preserve-list, the locked values (informational), the preset's
  `style_notes`, the photo metrics — and **revises** it on retry against the prior
  attempt's face-check. The writer produces the body **only**; it never sees
  `identity_instruction`, `negative_prompt` or the locked attributes as editable
  text. `services/prompt_builder.assemble_prompt` wraps the body deterministically
  — `identity (frozen) + body (sanitized) + locks (deterministic) + exclusions
  (frozen)` — so a lock wins over the body and the frozen blocks are untouchable.
  Every writer node degrades to the no-LLM fallback (the filled `prompt_structure`
  template + the fixed identity-emphasis line) when the writer is unavailable.
- **Edit-mode identity lock.** A one-per-photo VLM call (`extract_inventory`
  node, `InventoryExtractor` port) catalogues what the reference photo shows
  (pose, hands, accessories, clothing, hair, lighting, background) into
  `FaceProfile.inventory`. Edit steps assemble via
  `prompt_builder.assemble_edit_prompt`: a deterministic **LOCKED** enumeration
  (generic person lock + inventory + preserve-list + completed steps' `applied`
  phrases at their *new* values, minus the regions any step has edited), a
  single-change scope line ("the ONLY change allowed is …"), the writer's
  delta-only body, and a face-texture/integration block. The planner emits one
  change per step (only background/lighting/grade merge), ordered scene-first →
  clothing → face-adjacent last, each step with an `applied` phrase for the
  ledger. Edit mode also overrides the preset `aspect_ratio` with the source
  photo's nearest supported ratio — forcing a different ratio makes the model
  recompose the frame. An empty inventory degrades to the generic person lock;
  nothing in this path can fail a session.

## Presets

The preset library is the moat and is **not** in this public repo. The public
side ships only the preset **schema** (`schemas/presets.py`) and 2–3 demo presets
for tests (MIT). The real library is an external, independently semver-versioned
artifact the core loads at runtime — shipped as the private pip package
**`photocore-presets`** (assets read via `importlib.resources`). Local clone of
that repo lives at `../presets` (`git@github.com:ErikRusanov/Shoot-Craft-presets.git`).

`services/preset_matcher.py` resolves the source by config, no hardcoded path,
loads into memory at startup, indexes, and holds it immutably:

- `PRESET_SOURCE=package` (prod) → `importlib.resources.files("photocore_presets")`
- `PRESET_SOURCE=path` + `PRESET_LIBRARY_PATH=…` (dev/custom) → read that dir
- default (no config) → `presets/examples/` from this repo

**Preset schema v5 (writer pipeline, greedy budget).** A preset declares `mode`
(`generate`/`edit`/`both`), free-text `style_notes` for the writer, and per-slot
`policy` (`locked` = a deterministic non-negotiable attribute that wins over a
user delta, the conflict surfaced; `default` = the preset default, a user delta
wins). `prompt_structure` is **demoted** to the no-LLM fallback template — the
writer composes the body as the primary path; the builder assembles the frozen
blocks and locks around it. v5 dropped the `convergence` block: budget is spent
greedily (reserve-per-generation, stop when the next overdraws), not forecast
from a per-step attempt count, so a preset carries no generation-count estimate.

**Fallback convention (needs `photocore-presets >= 0.3.0`).** The id `default`
is the reserved fallback preset and the base **edit-mode** preset; its only
`applies_to.use_case` token is `default`, reserved and never used elsewhere.
`match()` excludes that token, so the fallback is *never* keyword-matched —
`resolve()` returns it only when no curated preset admits the request (i.e. for
an edit brief with no curated use-case). Its single `ask:true` slot (`scene`) is
the one **free-form** (no-enum) slot, fed from the brief; `assemble_prompt`
sanitizes the composed body (scene description only — attempts to edit the
face/identity or override the frozen blocks are rejected, the caller re-asks).
`PRESET_MIN_LIBRARY_VERSION` (default `0.8.0` — preset schema v5, dropping the
convergence block on top of the 0.7.0 pixel-for-pixel lock blocks) is enforced in
`package` mode at startup so a deploy can't silently lose the fallback or run a
stale contract.

`SessionState` must record `preset_id`, `preset_version` **and** `library_version`
(the package version) — otherwise a result can't be reproduced after a library
update. When the library outgrows a pip package (large/frequently-changing
assets, releases faster than deploys), move it to object storage and pull a
versioned tarball at startup via the same loader interface — later, not for MVP.

## Commits

Every commit subject starts with a type prefix: `(feat):`, `(fix):`, `(chore):`,
`(refactor):`, `(docs):`, `(test):`, `(perf):` — e.g. `(feat): add preset matcher`.

## Code style

- `async` on the whole I/O path.
- External dependencies (model, embedder, storage, Redis) strictly behind the
  `protocols` ports so tests can swap them.
- The contract is defined **only** in `schemas/`.
- No PII/biometrics in logs — only `session_key`/`face_key` and metrics.
- Comments explain **why**, not what.
