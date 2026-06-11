"""LLM-backed :class:`~protocols.classifier.UseCaseClassifier` via a cheap model.

Maps the user's free-text brief onto the library's curated ``use_case``
vocabulary with structured output (``response_format: json_schema``, ``strict``)
so the answer is always one of the allowed tokens or the reserved ``default``.
Its authority is exactly the port's: pick a token, nothing else — it never sees
preset content, prompts or identity blocks.

Failure policy mirrors the slot filler: any misbehavior — budget refusal,
transport failure, a 4xx, unparseable output — degrades to the deterministic
:class:`~services.classifier.TokenOverlapClassifier` instead of raising. The
session must never fail because classification did.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import structlog

from protocols.budget import BudgetMeter
from protocols.classifier import ClassifyResult, UseCaseClassifier
from schemas import PaidCallKind
from services.classifier import FALLBACK_USE_CASE, TokenOverlapClassifier
from services.connectors.openrouter_client import OpenRouterClient, parse_usage

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You route a user's photo request to one use-case category. You are given the "
    "allowed categories and the user's description. Choose exactly one category "
    "from the allowed list that best fits the request. If none fits, choose "
    f"{FALLBACK_USE_CASE!r}. Answer only with the chosen category token."
)


class _InvalidClassification(Exception):
    """Internal: the LLM's output failed validation — triggers the fallback."""


def _response_schema(use_cases: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"use_case": {"type": "string", "enum": [*use_cases, FALLBACK_USE_CASE]}},
        "required": ["use_case"],
        "additionalProperties": False,
    }


def _parse_use_case(body: dict[str, Any], allowed: set[str]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
        use_case = json.loads(content)["use_case"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise _InvalidClassification(f"unparseable classifier response: {exc!r}") from exc
    if not isinstance(use_case, str) or use_case not in allowed:
        raise _InvalidClassification("classifier returned a token outside the vocabulary")
    return use_case


class OpenRouterUseCaseClassifier(UseCaseClassifier):
    """Classifier that asks a cheap LLM, falling back to token overlap."""

    def __init__(
        self,
        client: OpenRouterClient,
        *,
        model: str,
        fallback: UseCaseClassifier | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._fallback = fallback if fallback is not None else TokenOverlapClassifier()

    async def classify(
        self, *, brief: str, use_cases: Sequence[str], meter: BudgetMeter | None = None
    ) -> ClassifyResult:
        if not brief.strip() or not use_cases:
            # Nothing to interpret — token overlap (or the fallback token) suffices.
            return await self._fallback.classify(brief=brief, use_cases=use_cases)
        # Reserve before the paid call; a refused budget degrades to the free
        # deterministic classifier rather than failing the session.
        reservation = None if meter is None else await meter.reserve(PaidCallKind.CLASSIFY)
        if meter is not None and reservation is None:
            log.warning("classifier_budget_exhausted")
            return await self._fallback.classify(brief=brief, use_cases=use_cases)
        try:
            body = await self._client.chat_completion(self._payload(brief, use_cases))
            use_case = _parse_use_case(body, {*use_cases, FALLBACK_USE_CASE})
            usage = parse_usage(body)
            if reservation is None:
                return ClassifyResult(use_case=use_case)
            cost = await reservation.settle(usage.cost if usage is not None else None)
            return ClassifyResult(use_case=use_case, usage=usage, cost=cost)
        # Deliberately broad: whatever the LLM path did wrong, degrade to the
        # deterministic classifier. The provider delivered no usable result, so
        # the reservation is refunded.
        except Exception as exc:
            if reservation is not None:
                await reservation.cancel()
            log.warning("classifier_llm_fallback", error=repr(exc))
            return await self._fallback.classify(brief=brief, use_cases=use_cases)

    def _payload(self, brief: str, use_cases: Sequence[str]) -> dict[str, Any]:
        user_message = {"allowed_use_cases": list(use_cases), "request": brief}
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
                    "name": "use_case",
                    "strict": True,
                    "schema": _response_schema(use_cases),
                },
            },
        }
