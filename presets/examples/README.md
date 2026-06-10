# Demo presets (MIT)

These 2–3 presets exist so the **public** core runs out of the box with no
private library installed. They are the default source (`PRESET_SOURCE=examples`)
and back the loader/schema tests.

They are **not** the curated library — that is the moat, shipped separately as the
private `photocore-presets` package (`PRESET_SOURCE=package`) or pointed at via
`PRESET_SOURCE=path`. Keep these generic and minimal; do not move real presets here.

`default.yaml` is the reserved **fallback** demo: id `default`, use_case `[default]`.
The matcher resolves to it when no other preset admits the request, and its single
`scene` slot is **free-form** (no enum) — the prompt builder sanitizes that text.

The shape is defined by `src/schemas/presets.py` — the canonical `Preset` schema.
