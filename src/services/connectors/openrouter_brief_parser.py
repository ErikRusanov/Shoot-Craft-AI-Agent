"""LLM-backed :class:`~protocols.brief_parser.BriefParser` via a cheap model.

Reads the user's free-text request into a :class:`BriefAnalysis` with structured
output (``response_format: json_schema``, ``strict``): the mode fork
(edit/generate), what to preserve, the changes (target + instruction each), the
conflicts, and the chosen ``use_case`` for a generate-mode curated preset.

Its authority is exactly the port's: analyze the request, nothing else — it never
sees preset content, prompts or identity blocks, and its output is re-validated
here. Failure policy mirrors the classifier and slot filler: any misbehavior —
budget refusal, transport failure, a 4xx, unparseable output, an out-of-vocabulary
use_case — degrades to the deterministic
:class:`~services.brief_parser.DeterministicBriefParser` instead of raising. The
session must never fail because parsing did. The parse reserves a ``CLASSIFY``
slot (it replaces that call), so pricing and the budget already cover it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import structlog

from protocols.brief_parser import BriefParser, ParseResult
from protocols.budget import BudgetMeter
from schemas import BriefAnalysis, Change, PaidCallKind
from services.brief_parser import DeterministicBriefParser
from services.connectors.openrouter_client import OpenRouterClient, parse_usage

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You analyze a user's photo request and return a structured reading. Decide the "
    "mode: 'edit' when the user wants to keep their existing photo and change only "
    "specific things (e.g. 'keep my face, replace the background with blue'); "
    "'generate' when they want a fresh styled photo for a target use case (e.g. a "
    "headshot, an avatar, a passport photo). For generate, pick the best-fitting "
    "use_case from the allowed list, or null if none fits; for edit, use_case is "
    "null. List 'preserve' (what must stay as in the original — pose, framing, "
    "clothing, setting; the face is always preserved implicitly). List 'changes', "
    "each a short target noun (background, lighting, clothing, accessory) plus the "
    "instruction in the user's intent. List 'conflicts': any request that tries to "
    "edit the person's face or identity itself, or asks for something impossible. "
    "Never restate the face as a change."
)


class _InvalidAnalysis(Exception):
    """Internal: the LLM's output failed validation — triggers the fallback."""


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["edit", "generate"]},
            "use_case": {"type": ["string", "null"]},
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
        "required": ["mode", "use_case", "preserve", "changes", "conflicts"],
        "additionalProperties": False,
    }


def _str_list(raw: object, field: str) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise _InvalidAnalysis(f"{field} is not a list of strings")
    return [x.strip() for x in raw if x.strip()]


def _parse_analysis(body: dict[str, Any], allowed: set[str]) -> BriefAnalysis:
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise _InvalidAnalysis(f"unparseable parser response: {exc!r}") from exc
    if not isinstance(parsed, dict):
        raise _InvalidAnalysis("parser response is not an object")

    mode = parsed.get("mode")
    if mode not in {"edit", "generate"}:
        raise _InvalidAnalysis(f"mode {mode!r} is not edit/generate")

    use_case = parsed.get("use_case")
    if use_case is not None:
        if not isinstance(use_case, str) or use_case not in allowed:
            raise _InvalidAnalysis("use_case is outside the curated vocabulary")
    if mode == "edit":
        use_case = None  # an edit never targets a curated use case

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
        mode=mode,
        use_case=use_case,
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
        use_case: str | None,
        use_cases: Sequence[str],
        meter: BudgetMeter | None = None,
    ) -> ParseResult:
        if not brief.strip():
            # Nothing to analyze — the deterministic parse covers it for free.
            return await self._fallback.parse(brief=brief, use_case=use_case, use_cases=use_cases)
        # Reserve before the paid call; a refused budget degrades to the free
        # deterministic parse rather than failing the session.
        reservation = None if meter is None else await meter.reserve(PaidCallKind.CLASSIFY)
        if meter is not None and reservation is None:
            log.warning("brief_parser_budget_exhausted")
            return await self._fallback.parse(brief=brief, use_case=use_case, use_cases=use_cases)
        try:
            body = await self._client.chat_completion(self._payload(brief, use_case, use_cases))
            analysis = _parse_analysis(body, set(use_cases))
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
            return await self._fallback.parse(brief=brief, use_case=use_case, use_cases=use_cases)

    def _payload(
        self, brief: str, use_case: str | None, use_cases: Sequence[str]
    ) -> dict[str, Any]:
        user_message = {
            "allowed_use_cases": list(use_cases),
            "known_use_case": use_case,
            "request": brief,
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
                    "name": "brief_analysis",
                    "strict": True,
                    "schema": _response_schema(),
                },
            },
        }
