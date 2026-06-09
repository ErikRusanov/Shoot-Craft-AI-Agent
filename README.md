# Shoot-Craft-AI-Agent
AI photoshoot agent — your photos in, studio-quality shots with your real face out. No prompting needed

## Presets & the private library

The curated preset library is the moat and lives in a **separate** repo,
`photocore-presets` (`../presets` locally). This public core ships only the
`Preset` **schema** (`src/schemas/presets.py`, the canonical contract) and a few
MIT **demo** presets (`presets/examples/`). The core has **no dependency** on the
private package — it discovers presets at runtime via config.

`services/preset_matcher.py` resolves the source from `PRESET_SOURCE`:

| `PRESET_SOURCE` | Reads from | `library_version` | Use |
| --------------- | ---------- | ----------------- | --- |
| `examples` (default) | `presets/examples/` | `examples` | public core runs out of the box |
| `path` | `PRESET_LIBRARY_PATH` dir | `path:<dir>` (not reproducible) | local dev against `../presets/src/library` |
| `package` | installed `PRESET_PACKAGE` | the dist version (reproducible) | prod / private image |

### Local dev against the private library

```bash
# Option A — read the sibling checkout directly (no install, edits live):
#   in .env:
#     PRESET_SOURCE=path
#     PRESET_LIBRARY_PATH=../presets/src/library

# Option B — exercise the real prod (package) path:
make presets-dev          # editable-install ../presets into this venv
#   in .env: PRESET_SOURCE=package
#   NOTE: `make sync` prunes the editable install — re-run `make presets-dev` after.
```

### Docker

Two images, mirroring the boundary:

- **Public base** (`./Dockerfile`) — core + demo presets, `PRESET_SOURCE=examples`.
  ```bash
  docker build -t photocore-base:latest .
  ```
- **Private** (`../presets/Dockerfile`) — builds the library wheel from source,
  then installs it onto the base; `library_version` = the wheel version.
  ```bash
  docker build -t photocore-private:0.2.0 \
      --build-arg BASE_IMAGE=photocore-base:latest ../presets
  ```
  (`make presets-build` is the standalone "produce a wheel artifact" target, for
  publishing to a private index or CI — not needed for the Docker build above.)

The private library never enters the public repo or the public image; it is
layered on only in the private image.
