"""LLM-backed :class:`~protocols.brief_parser.BriefParser` via a cheap model.

Reads the user's free-text request into a :class:`BriefAnalysis` with structured
output (``response_format: json_schema``, ``strict``): what to preserve, the
changes (target + instruction each), and the conflicts. Mode is always ``edit``.

Its authority is exactly the port's: analyze the request, nothing else — it never
sees preset content, prompts or identity blocks, and its output is re-validated
here. Failure policy mirrors the classifier and slot filler: any misbehavior —
budget refusal, transport failure, a 4xx, unparseable output — degrades to the
deterministic :class:`~services.brief_parser.DeterministicBriefParser` instead of
raising. The session must never fail because parsing did. The parse reserves a
``CLASSIFY`` slot (it replaces that call), so pricing and the budget already
cover it.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from protocols.brief_parser import BriefParser, ParseResult
from protocols.budget import BudgetMeter
from schemas import BriefAnalysis, Change, PaidCallKind
from services.brief_parser import DeterministicBriefParser
from services.connectors.openrouter_client import OpenRouterClient, parse_usage

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You analyze a user's photo-editing request and return a structured reading.\n"
    "\n"
    "The pipeline always works in edit mode — the user's own photo is the starting "
    "point and only the named changes are applied. Your job is to extract:\n"
    "\n"
    "- 'preserve': what must stay exactly as in the original photo — pose, framing, "
    "clothing, setting; the face is always preserved implicitly so never list it.\n"
    "- 'changes': each a short target noun (background, lighting, clothing, accessory, "
    "color_grade) plus the instruction in the user's own intent. Use one change per "
    "distinct region — do not merge unrelated targets.\n"
    "- 'conflicts': any request that tries to edit the person's face or identity "
    "itself, or asks for something impossible. Never restate the face as a change.\n"
    "\n"
    "If the user mentions a general quality goal ('professional look', 'studio "
    "quality') AND specific changes, include the specific changes and treat the "
    "quality goal as style context in 'preserve'. Never invent changes the user did "
    "not ask for."
)


class _InvalidAnalysis(Exception):
    """Internal: the LLM's output failed validation — triggers the fallback."""


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "preserve": {"type": "array", "items": {"type": "string"}},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "instruction": {"type": "string"},
                    },
                    "required": ["target", "instruction"],
                    "additionalProperties": False,
                },
            },
            "conflicts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["preserve", "changes", "conflicts"],
        "additionalProperties": False,
    }


def _str_list(raw: object, field: str) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise _InvalidAnalysis(f"{field} is not a list of strings")
    return [x.strip() for x in raw if x.strip()]


def _parse_analysis(body: dict[str, Any]) -> BriefAnalysis:
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise _InvalidAnalysis(f"unparseable parser response: {exc!r}") from exc
    if not isinstance(parsed, dict):
        raise _InvalidAnalysis("parser response is not an object")

    raw_changes = parsed.get("changes")
    if not isinstance(raw_changes, list):
        raise _InvalidAnalysis("changes is not a list")
    changes: list[Change] = []
    for item in raw_changes:
        if not isinstance(item, dict):
            raise _InvalidAnalysis("a change is not an object")
        target, instruction = item.get("target"), item.get("instruction")
        if not isinstance(target, str) or not isinstance(instruction, str):
            raise _InvalidAnalysis("a change is missing target/instruction")
        if target.strip() and instruction.strip():
            changes.append(Change(target=target.strip(), instruction=instruction.strip()))

    return BriefAnalysis(
        preserve=_str_list(parsed.get("preserve"), "preserve"),
        changes=changes,
        conflicts=_str_list(parsed.get("conflicts"), "conflicts"),
    )


class OpenRouterBriefParser(BriefParser):
    """BriefParser that asks a cheap LLM, falling back to the deterministic parse."""

    def __init__(
        self,
        client: OpenRouterClient,
        *,
        model: str,
        fallback: BriefParser | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._fallback = fallback if fallback is not None else DeterministicBriefParser()

    async def parse(
        self,
        *,
        brief: str,
        meter: BudgetMeter | None = None,
    ) -> ParseResult:
        if not brief.strip():
            # Nothing to analyze — the deterministic parse covers it for free.
            return await self._fallback.parse(brief=brief)
        # Reserve before the paid call; a refused budget degrades to the free
        # deterministic parse rather than failing the session.
        reservation = None if meter is None else await meter.reserve(PaidCallKind.CLASSIFY)
        if meter is not None and reservation is None:
            log.warning("brief_parser_budget_exhausted")
            return await self._fallback.parse(brief=brief)
        try:
            body = await self._client.chat_completion(self._payload(brief))
            analysis = _parse_analysis(body)
            usage = parse_usage(body)
            if reservation is None:
                return ParseResult(analysis=analysis)
            cost = await reservation.settle(usage.cost if usage is not None else None)
            return ParseResult(analysis=analysis, usage=usage, cost=cost)
        # Deliberately broad: whatever the LLM path did wrong, degrade to the
        # deterministic parse. The provider delivered no usable result, so the
        # reservation is refunded.
        except Exception as exc:
            if reservation is not None:
                await reservation.cancel()
            log.warning("brief_parser_llm_fallback", error=repr(exc))
            return await self._fallback.parse(brief=brief)

    def _payload(self, brief: str) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"request": brief}, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "brief_analysis",
                    "strict": True,
                    "schema": _response_schema(),
                },
            },
        }
