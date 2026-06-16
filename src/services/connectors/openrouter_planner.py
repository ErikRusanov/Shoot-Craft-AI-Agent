"""LLM-backed :class:`~protocols.planner.StepPlanner`.

Decides the judgments the deterministic plan cannot: the exact step order, which
same-region scene tweaks merge into one edit, how each instruction resolves
against the photo inventory ("replace the existing earbud…", not "add a
headset"), and the ``applied`` phrase later steps lock. Structured output
(``response_format: json_schema``, ``strict``) returns the steps; every input
target must be covered exactly once, validated here, so the planner can neither
drop nor invent a change.

The planner also reconciles the brief against the photo inventory. The brief
parser sees only the user's text, so anything the user describes as their current
state (the preserve list) may differ from what the photo actually shows. Any
discrepancy is an implicit change — the planner discovers and plans it alongside
the explicit changes. This is why inventory + a non-empty preserve list can
trigger an LLM call even with only one explicit change.

Failure policy mirrors the other ports: any misbehavior degrades to the
deterministic plan. The planner reserves a ``SLOT_FILL`` slot (a cheap auxiliary
call), so pricing and the budget already cover it.
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

_SYSTEM_PROMPT = (
    "You plan photo-edit steps for a reference-conditioned image editor that applies "
    "ONE change at a time, chaining outputs: step N+1 edits the output of step N. "
    "Identity drift compounds with every pass, so minimize passes and risk per pass.\n"
    "\n"
    "PHASE 1 — RECONCILE THE BRIEF AGAINST THE PHOTO:\n"
    "You receive two kinds of input:\n"
    "  • explicit_changes — changes the user explicitly requested.\n"
    "  • preserve_list — what the user described as their current photo (pose, setting,\n"
    "    clothing, etc.) that they want kept.\n"
    "  • photo_inventory — what the photo actually shows.\n"
    "Compare each preserve_list item against the matching field in photo_inventory. "
    "If the user's description of their photo differs from what the photo actually "
    "shows, that discrepancy is an IMPLICIT change to add to the plan.\n"
    "Examples:\n"
    "  • preserve says 'city background setting', inventory shows 'residential interior "
    "with white shelving' → add a step: change background to a city/urban setting.\n"
    "  • preserve says 'casual wear (jeans and jacket)', inventory shows 'beige t-shirt' "
    "→ add a step: replace the beige t-shirt with jeans and a casual jacket.\n"
    "Do NOT add an implicit change when the preserve item is a generic instruction "
    "('face and identity', 'framing and composition') rather than a description of a "
    "specific visual element, or when the description and inventory say the same thing "
    "in different words. Only flag real, clear visual discrepancies.\n"
    "\n"
    "PHASE 2 — ORDER ALL CHANGES INTO STEPS:\n"
    "Combine explicit_changes and any implicit changes from phase 1 into an ordered plan.\n"
    "Rules:\n"
    "- One change per step by default. Merge ONLY when changes target the same scene "
    "region: background, ambient lighting, and overall color grade may merge into one "
    "step. Nothing else merges — never merge clothing with accessories; never merge "
    "anything near the face.\n"
    "- Order steps from least to most identity-sensitive: (1) scene-level changes "
    "(background, setting, lighting, color grade) first; (2) clothing and body-level "
    "next; (3) accessories and anything close to the face (glasses, earrings, hat) last.\n"
    "- Make each instruction self-contained and concrete. Resolve references against "
    "photo_inventory: write 'replace the existing beige t-shirt with dark jeans and a "
    "casual blazer jacket', not 'change clothing'. Name colors, materials, and "
    "placement explicitly.\n"
    "- For each step write 'applied': a short noun phrase naming the changed attribute "
    "at its NEW value once the step is done (e.g. 'the new dark jeans and blazer '). "
    "Later steps will lock this phrase as untouchable.\n"
    "- Every explicit_change target must appear in exactly one step. Implicit targets "
    "discovered in phase 1 are additional steps. Do not drop any target."
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


def _parse_steps(
    body: dict[str, Any],
    analysis: BriefAnalysis,
    *,
    allow_extra_targets: bool = False,
) -> list[EditStep]:
    try:
        content = body["choices"][0]["message"]["content"]
        raw_steps = json.loads(content)["steps"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise _InvalidPlan(f"unparseable planner response: {exc!r}") from exc
    if not isinstance(raw_steps, list) or not raw_steps:
        raise _InvalidPlan("planner returned no steps")

    explicit_targets = {c.target for c in analysis.changes}
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
        # Extra targets (not in the brief) are only allowed when an inventory
        # was provided — they are implicit changes discovered by reconciliation.
        # Without an inventory the planner has nothing to reconcile against, so
        # an extra target is a hallucination and the plan falls back.
        if not allow_extra_targets and not set(clean) <= explicit_targets:
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

    # All explicit targets must be covered; extra steps for implicit (discovered)
    # changes are permitted beyond the explicit set.
    if not explicit_targets <= covered:
        raise _InvalidPlan(
            f"explicit changes not covered by any step: {explicit_targets - covered!r}"
        )
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
        has_inventory = inventory is not None and not inventory.is_empty()
        needs_reconciliation = has_inventory and bool(analysis.preserve)
        if len(analysis.changes) < 2 and not needs_reconciliation:
            # Single explicit change and no inventory to reconcile against —
            # deterministic one-step plan, free.
            return await self._fallback.plan(analysis=analysis)
        # Reserve before the paid call; a refused budget degrades to the
        # deterministic plan rather than failing the session.
        reservation = None if meter is None else await meter.reserve(PaidCallKind.SLOT_FILL)
        if meter is not None and reservation is None:
            log.warning("planner_budget_exhausted")
            return await self._fallback.plan(analysis=analysis)
        try:
            body = await self._client.chat_completion(self._payload(analysis, inventory))
            steps = _parse_steps(
                body,
                analysis,
                allow_extra_targets=has_inventory,
            )
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
        user_message: dict[str, Any] = {
            "explicit_changes": [
                {"target": c.target, "instruction": c.instruction} for c in analysis.changes
            ],
            # What the user described as their current photo — compared against
            # photo_inventory in phase 1 to discover implicit changes.
            "preserve_list": analysis.preserve,
            # What the photo actually shows — the ground truth for reconciliation.
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
