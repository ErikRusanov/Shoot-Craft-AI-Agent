"""LLM-backed :class:`~protocols.planner.StepPlanner`.

Decides the judgments the deterministic plan cannot: the exact step order, which
same-region scene tweaks merge into one edit, how each instruction resolves
against the photo inventory ("replace the existing earbud…", not "add a
headset"), and the ``applied`` phrase later steps lock. Structured output
(``response_format: json_schema``, ``strict``) returns the steps; every input
target must be covered exactly once, validated here, so the planner can neither
drop nor invent a change.

Only edit-mode briefs with two or more changes are worth a call — a generate
brief or a single change is the deterministic single/one-step plan, free. Failure
policy mirrors the other ports: any misbehavior degrades to the deterministic
plan. The planner reserves a ``SLOT_FILL`` slot (a cheap auxiliary call), so
pricing and the budget already cover it.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from protocols.budget import BudgetMeter
from protocols.planner import PlanResult, StepPlanner
from schemas import BriefAnalysis, EditStep, PaidCallKind, PhotoInventory
from services.connectors.openrouter_client import OpenRouterClient, parse_usage
from services.planner import DeterministicStepPlanner

log = structlog.get_logger(__name__)

# One change per step is the rule, not the exception: every pass through the
# editor risks the face, but a narrow pass regenerates less — the lock block
# around each step can only say "the ONLY change allowed" when the step really
# is one change.
_SYSTEM_PROMPT = (
    "You order a list of requested photo-edit changes into steps for a "
    "reference-conditioned image editor that applies ONE edit at a time and chains "
    "results: the output of step 1 is the input of step 2. Identity drift compounds "
    "with every pass, so minimize the number of passes and the risk per pass.\n"
    "Rules:\n"
    "- One change per step by default. Merge changes into a single step ONLY when "
    "they describe the same scene region: background, ambient lighting and overall "
    "color grade may merge into one step. Nothing else merges — never merge a "
    "clothing change with an accessory change, and never merge anything with a "
    "change near the face.\n"
    "- Order steps from least to most identity-sensitive: (1) scene-level changes "
    "(background, setting, lighting, color grade) first; (2) clothing and "
    "body-level changes next; (3) accessories and anything close to the face "
    "(glasses, earrings, a headset, a hat) last.\n"
    "- Make each instruction self-contained and concrete. Resolve references "
    "against the photo inventory when provided: write 'replace the existing white "
    "earbud in the left ear with a black over-ear headset microphone', not 'add a "
    "headset'. Name colors, materials and placement explicitly.\n"
    "- For each step also write 'applied': a short noun phrase naming the changed "
    "attribute at its NEW value once the step is done (e.g. 'the new plain white "
    "crew-neck t-shirt'). Later steps will lock this phrase as untouchable.\n"
    "- Each step gets a short title, the instruction, the list of target names it "
    "covers, and 'applied'. Every input target must appear in exactly one step. "
    "Do not invent or drop targets."
)


class _InvalidPlan(Exception):
    """Internal: the LLM's output failed validation — triggers the fallback."""


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "instruction": {"type": "string"},
                        "targets": {"type": "array", "items": {"type": "string"}},
                        "applied": {"type": "string"},
                    },
                    "required": ["title", "instruction", "targets", "applied"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }


def _parse_steps(body: dict[str, Any], analysis: BriefAnalysis) -> list[EditStep]:
    try:
        content = body["choices"][0]["message"]["content"]
        raw_steps = json.loads(content)["steps"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise _InvalidPlan(f"unparseable planner response: {exc!r}") from exc
    if not isinstance(raw_steps, list) or not raw_steps:
        raise _InvalidPlan("planner returned no steps")

    wanted = {c.target for c in analysis.changes}
    covered: set[str] = set()
    steps: list[EditStep] = []
    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            raise _InvalidPlan("a step is not an object")
        title, instruction, targets = (
            item.get("title"),
            item.get("instruction"),
            item.get("targets"),
        )
        if not isinstance(title, str) or not title.strip():
            raise _InvalidPlan("a step has no title")
        if not isinstance(instruction, str) or not instruction.strip():
            raise _InvalidPlan("a step has no instruction")
        if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
            raise _InvalidPlan("a step's targets are not strings")
        clean = [t.strip() for t in targets if t.strip()]
        if not set(clean) <= wanted:
            raise _InvalidPlan("a step covers a target not in the brief")
        if covered & set(clean):
            raise _InvalidPlan("a target is covered by more than one step")
        covered.update(clean)
        # `applied` is advisory: an empty/missing phrase degrades to the loop's
        # generic fallback, never invalidates an otherwise sound plan.
        applied = item.get("applied")
        steps.append(
            EditStep(
                n=i,
                title=title.strip(),
                instruction=instruction.strip(),
                targets=clean,
                applied=applied.strip() if isinstance(applied, str) else "",
            )
        )

    if covered != wanted:
        raise _InvalidPlan("not every change is covered by a step")
    return steps


class OpenRouterStepPlanner(StepPlanner):
    """StepPlanner that asks a cheap LLM, falling back to the deterministic plan."""

    def __init__(
        self,
        client: OpenRouterClient,
        *,
        model: str,
        fallback: StepPlanner | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._fallback = fallback if fallback is not None else DeterministicStepPlanner()

    async def plan(
        self,
        *,
        analysis: BriefAnalysis,
        inventory: PhotoInventory | None = None,
        meter: BudgetMeter | None = None,
    ) -> PlanResult:
        if analysis.mode == "generate" or len(analysis.changes) < 2:
            # Nothing to decide — the deterministic single/one-step plan, free.
            return await self._fallback.plan(analysis=analysis)
        # Reserve before the paid call; a refused budget degrades to the
        # deterministic plan rather than failing the session.
        reservation = None if meter is None else await meter.reserve(PaidCallKind.SLOT_FILL)
        if meter is not None and reservation is None:
            log.warning("planner_budget_exhausted")
            return await self._fallback.plan(analysis=analysis)
        try:
            body = await self._client.chat_completion(self._payload(analysis, inventory))
            steps = _parse_steps(body, analysis)
            usage = parse_usage(body)
            if reservation is None:
                return PlanResult(steps=steps)
            cost = await reservation.settle(usage.cost if usage is not None else None)
            return PlanResult(steps=steps, usage=usage, cost=cost)
        # Deliberately broad: whatever the LLM path did wrong, degrade to the
        # deterministic plan. The provider delivered no usable result, refund.
        except Exception as exc:
            if reservation is not None:
                await reservation.cancel()
            log.warning("planner_llm_fallback", error=repr(exc))
            return await self._fallback.plan(analysis=analysis)

    def _payload(self, analysis: BriefAnalysis, inventory: PhotoInventory | None) -> dict[str, Any]:
        user_message = {
            "changes": [
                {"target": c.target, "instruction": c.instruction} for c in analysis.changes
            ],
            # What the photo actually shows, so instructions resolve concretely;
            # null when no inventory was extracted.
            "photo_inventory": inventory.model_dump(exclude={"schema_v"})
            if inventory is not None and not inventory.is_empty()
            else None,
        }
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_message, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "step_plan",
                    "strict": True,
                    "schema": _response_schema(),
                },
            },
        }
