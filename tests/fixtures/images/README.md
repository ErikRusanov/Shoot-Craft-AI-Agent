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
| `face_a.jpg` | One clear, sharp, well-lit, **near-frontal** face of person **A** (head turned ≤ ~35° — a three-quarter portrait fails the pose check). Frame ≥ 512px on the short side, the face bbox ≥ ~128px. Small faces in the background are fine; another face of comparable size is not. This is the "passes the gate" anchor. |
| `face_b.jpg` | A **different person** — used to assert two identities embed apart. Only needs a detectable face; the gate is not applied to it. |

Any selfie-quality photo works. JPEG or anything pillow decodes; the names
must match exactly.

Degraded variants (blur, low light) are produced **programmatically** from
`face_a.jpg` inside the tests — don't add them here. Fully synthetic cases
(uniform fill = no face, generated noise) need no fixtures at all and never
skip.
