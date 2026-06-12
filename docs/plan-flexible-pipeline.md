# Plan: flexible brief-driven pipeline (agentic prompt writer)

Status: approved, not started. Implement stages in order; each stage is one
commit that keeps tests green. This document is self-contained — it carries the
diagnosis, the decisions already made with the owner, and the stage-by-stage
plan.

## Why (diagnosis)

Real failure case: brief *"keep my face the same, but replace the background
with blue"* produced a frontal grey-backdrop avatar — wrong pose, wrong
background, the word "blue" silently lost. Root causes, all architectural:

1. **`classify` collapses the brief to one use-case token.** The brief's
   constraints ("keep X as is") and deltas ("change Y to blue") are discarded;
   only "avatar" survives. The edit-framed `default` preset (which would have
   done the right thing) is unreachable whenever any curated token vaguely
   matches.
2. **Presets are templates of a target image, not operations.** `demo_avatar`'s
   `prompt_structure` hard-codes "looking at the camera" — the model gets
   "generate an avatar", not "edit this photo". Delta-driven briefs (change one
   thing, keep the rest) have no path.
3. **Closed enum slots are lossy and silent.** `background` admits two greys;
   the slot filler (strict structured output) snapped "blue" to grey without
   telling anyone.
4. **The only quality signal is embedding cosine**, which is pose- and
   background-blind: similarity 0.74 = "passed" while 0 of 2 brief items were
   fulfilled. Retries only append a fixed identity-emphasis line and decay
   temperature — they cannot fix "wrong background" because nothing measures it.
5. **`Plan` is a cost estimate, not a plan.** A complex brief (background +
   lighting + t-shirt + microphone) has nowhere to decompose into steps, even
   though the generator is reference-conditioned edit and chains naturally.

## Decisions already made (do not re-litigate)

- **The Prompt Writer is the agent's core.** An LLM composes the prompt body
  per situation (mode, step, preserve-list, photo metrics, prior-attempt
  feedback). Prompts are NOT glued from preset template pieces as the primary
  path.
- **Frozen forever:** `identity_instruction`, `negative_prompt`, and locked
  attributes. They are assembled into the final prompt deterministically,
  around the writer's body — the writer never sees them as editable text.
  This AMENDS the CLAUDE.md rule "prompt structure is frozen" (update CLAUDE.md
  in stage 8).
- **Rigidity lives in constraints, not string templates.** A passport-style
  preset declares locked attributes ("background pure white", "frontal pose");
  they win over user deltas, and the conflict is surfaced to the user
  explicitly — never silently overridden in either direction.
- **`prompt_structure` is demoted, not deleted:** it becomes the deterministic
  no-LLM fallback template. Every LLM node must degrade to today's behavior
  without an LLM (same convention as classifier / slot_filler today).
- **ON HOLD (owner decision):** the VLM compliance/critique check ("was the
  requested change applied; did anything in `preserve` drift"). Build the
  `revise()` hook on the writer port now with only facecheck feedback
  (similarity + verdict); the VLM critique plugs into `feedback` later without
  re-architecture.
- **No point-fixes for single cases** (e.g. "teach the classifier this one
  brief", "drop one enum"). Architectural solutions only.

## Stage 1 — schemas (whole contract, one commit)

- `schemas/brief.py` (new): `BriefAnalysis` with its own `schema_v`:
  - `mode: Literal["edit", "generate"]` — delta-driven vs target-driven;
  - `use_case: str | None`;
  - `preserve: list[str]` (e.g. face, pose, framing, clothing);
  - `changes: list[Change]`, `Change = {target: str, instruction: str}`;
  - `conflicts: list[str]` — asks that contradict locked attributes or try to
    edit the face; surfaced to the user, never silently dropped.
- `schemas/presets.py` → `schema_v: 4`:
  - `mode: Literal["generate", "edit", "both"]`;
  - `style_notes: str` — free-text style guidance fed to the writer;
  - slots gain `policy: Literal["locked", "default"]` (locked = deterministic,
    non-negotiable; default = preset default, user delta wins);
  - `prompt_structure` kept as the fallback template.
- `schemas/state.py` (bump `schema_v`):
  - `Plan.steps: list[EditStep]`,
    `EditStep = {n, title, instruction, targets: list[str], status, result_ref}`;
  - `Iteration.step_n: int`;
  - `SessionState.brief_analysis: BriefAnalysis | None`.
- No data migration: state is TTL-bound in Redis; version-gate on read.

## Stage 2 — brief parser (replaces classify)

- `services/brief_parser.py`: port + deterministic fallback;
  `services/connectors/openrouter_brief_parser.py`: LLM impl. One call returns
  `BriefAnalysis`, replacing the classify LLM call.
- Deterministic fallback = exactly today's behavior: token-overlap over
  use-case tokens; curated preset matches → `mode=generate` with that
  use_case; nothing matches → `mode=edit` with the whole brief as a single
  change. Today's `classifier.py` is absorbed into this fallback.

## Stage 3 — Prompt Writer + prompt_builder refactor (the heart)

- Port `PromptWriter`:
  - `compose(step, constraints, photo_metrics) -> body` — writes the scene/edit
    body only; receives the preserve-list, the step's changes, locked attribute
    values (informational), `style_notes`, photo metrics;
  - `revise(prev_body, feedback) -> body` — feedback is facecheck-only for now
    (similarity, verdict); the current `IDENTITY_EMPHASIS` + temperature decay
    becomes its deterministic fallback.
- `services/connectors/openrouter_prompt_writer.py` — structured output: body
  text only, never identity/negatives.
- `prompt_builder.py` changes role: from "fill the preset template" to
  "assemble `identity (frozen) + body (writer) + locks (deterministic) +
  exclusion clause (frozen)`", run the existing injection/sanitization patterns
  over the body, hash. `prompt_text`/`prompt_hash` in `Iteration` keep working
  unchanged (reproducibility preserved).

## Stage 4 — step planner

- `services/planner.py`: cuts `changes` into ordered steps — compatible deltas
  merge (background + lighting), independent ones split (t-shirt; microphone).
  LLM impl + deterministic fallback "one change = one step".
- `estimator.py`: forecast = Σ over steps of `expected_generations`. If the
  budget can't fit all steps, the planner trims the tail and records that in
  the plan explicitly — no silent caps.
- `mode=generate` is the degenerate case: a single step.

## Stage 5 — generation loop over steps

- Outer loop over `plan.steps`; inner loop = today's retry loop within a step.
- **Chaining:** the working image for step N+1 is step N's best result. The
  **identity anchor is always the crop of the original photo**, attached on
  every step (otherwise face drift accumulates along the chain). Facecheck of
  every iteration is against the ORIGINAL embedding, never the previous step.
- Keep-best per step. Terminal delivery = the last completed step's best
  result; accounting states explicitly which steps completed and which didn't.
  A partially completed chain is a valid result, not a failure.
- Budget mechanics (reserve → settle → refund) unchanged, just scoped inside a
  step.

## Stage 6 — graph

- `classify` → `parse_brief`; `match_fill` → `resolve_constraints` (resolve
  preset by mode + use_case, check `changes` against locks → conflicts);
  `plan` → `plan_steps`.
- `ask` is reframed: ask when the parser found nothing actionable, on
  conflicts, and for the ask-slot of generate-mode presets. The reask loop and
  `MAX_REASKS` stay.
- `approve` shows the step plan (that is what the user approves). FSM skeleton
  and interrupt semantics (re-execute node from top on resume, no pre-interrupt
  publishes) are untouched.

## Stage 7 — events & contract

- `PlanEvent` carries steps; new `StepStartedEvent` / `StepResultEvent` for
  chain progress; `DoneEvent` gets per-step accounting. Bump `schema_v` in
  `events.py` / `contract.py`. SSE mechanics unchanged.

## Stage 8 — presets & docs

- `presets/examples/`: `demo_avatar` v2 (mode: generate, locked/default
  policies instead of the enum cage), `default` becomes the base edit-mode
  preset, add a third passport-style demo with real locks (the rigid mode is
  tested on it).
- `../presets` (private library): migrate to preset schema 4 — major version
  bump of `photocore-presets`; raise `PRESET_MIN_LIBRARY_VERSION`.
- CLAUDE.md: rewrite the "Prompt adaptation" rule per the decisions above.
  Update README / docs diagrams (`docs/fsm.md`, `docs/architecture.md`).

## Golden test scenarios (built up across stages)

1. "Keep the face, blue background" → `mode=edit`, one step, prompt contains
   the blue backdrop, no "looking at the camera", `preserve` contains pose.
2. "Passport photo with a green background" → `mode=generate`, lock wins,
   the conflict is visible in the plan.
3. "Background + lighting + t-shirt + microphone" → 2–3 steps, per-step
   estimate, chained references, facecheck vs the original.
4. Every LLM node degrades to today's behavior when the LLM is unavailable.
5. Injection sanitization on the writer's output; a locked attribute cannot be
   overridden by the prompt body.

## Commit order

One commit per stage, tests green at every point: schemas land version-gated
alongside the old ones; services arrive with fallbacks; the graph switches
last; the old path is removed after the switch.
