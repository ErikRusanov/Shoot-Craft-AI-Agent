# Image fixtures (local only, never committed)

Real-people photos are biometric test data, so this directory ships empty: the
`.gitignore` here keeps everything but the docs out of the repo. Tests that
need these files **skip with a clear message** when they are absent — CI stays
green without them; drop the files in locally to run the full suite.

The InsightFace weights are a separate prerequisite for the same tests: run
`make models` once (see `INSIGHTFACE_ROOT` in `.env.example`).

## Files to provide

| file | what it must contain |
| --- | --- |
| `face_a.jpg` | One clear, sharp, frontal, well-lit face of person **A**. ≥ 512px on the short side, face filling a good part of the frame (≥ ~10% by area). This is the "passes the gate" anchor. |
| `face_b.jpg` | Same requirements, but a **different person** — used to assert two identities embed apart. |

Any selfie-quality photo works. JPEG or anything pillow decodes; the names
must match exactly.

Degraded variants (blur, low light) are produced **programmatically** from
`face_a.jpg` inside the tests — don't add them here. Fully synthetic cases
(uniform fill = no face, generated noise) need no fixtures at all and never
skip.
