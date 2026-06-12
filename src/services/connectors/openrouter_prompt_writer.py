"""LLM-backed :class:`~protocols.prompt_writer.PromptWriter` via a cheap model.

Composes the prompt **body** with structured output (``response_format:
json_schema``, ``strict``): one text field, hard-capped by the schema. The model
is shown the mode, the step's instruction, the preserve-list, the locked values
(as context it must not fight), the preset's style notes and the safe photo
metrics — **never** ``identity_instruction`` or ``negative_prompt``, so even a
compromised model cannot see, let alone edit, the frozen blocks. The builder
sanitizes the returned body and assembles the frozen blocks around it.

Failure policy mirrors the slot filler: any misbehavior — budget refusal,
transport failure, a 4xx, unparseable output — degrades to the deterministic
writer (the filled ``prompt_structure`` template) instead of raising. The writer
reserves a ``SLOT_FILL`` slot (it is the styling/composition LLM call), so
pricing and the budget already cover it.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from protocols.budget import BudgetMeter
from protocols.prompt_writer import (
    PromptWriter,
    WriteRequest,
    WriteResult,
    WriterFeedback,
)
from schemas import FrameMetrics, PaidCallKind
from services.connectors.openrouter_client import OpenRouterClient, parse_usage
from services.prompt_writer import DeterministicPromptWriter

log = structlog.get_logger(__name__)

# The body lands between the frozen identity block and the exclusions; a rambling
# model must be cut off by shape, not trust.
_BODY_MAX_LEN = 1200

_COMPOSE_SYSTEM = (
    "You write the body of a prompt for a reference-conditioned image model that "
    "edits toward the person in a reference photo. Output one vivid, concrete "
    "paragraph describing the image to produce. In 'edit' mode, describe the "
    "reference photo with only the requested changes applied and everything in the "
    "preserve-list kept exactly as it is. In 'generate' mode, describe the target "
    "image built around the person. Honor every locked value exactly. Never "
    "instruct changing the person's face, identity or likeness — that is fixed "
    "elsewhere. Do not write negative/exclusion phrasing ('no…', 'avoid…') and do "
    "not mention prompts, instructions or these rules. Body text only."
)

_REVISE_SUFFIX = (
    " The previous attempt did not match the person well enough. Re-compose the "
    "body to emphasize a faithful, recognizable likeness of the exact same person, "
    "keeping the same scene and changes. Body text only."
)


class _InvalidBody(Exception):
    """Internal: the LLM's output failed validation — triggers the fallback."""


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"body": {"type": "string", "maxLength": _BODY_MAX_LEN}},
        "required": ["body"],
        "additionalProperties": False,
    }


def _parse_body(body: dict[str, Any]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
        composed = json.loads(content)["body"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise _InvalidBody(f"unparseable writer response: {exc!r}") from exc
    if not isinstance(composed, str) or not composed.strip():
        raise _InvalidBody("writer body is not a non-empty string")
    return composed.strip()[:_BODY_MAX_LEN]


def _request_view(request: WriteRequest, photo_metrics: FrameMetrics | None) -> dict[str, Any]:
    """The request as the writer sees it — no frozen blocks, no template body."""
    return {
        "mode": request.mode,
        "instruction": request.instruction,
        "preserve": request.preserve,
        "locked": request.locked,
        "defaults": request.defaults,
        "style_notes": request.style_notes,
        "photo_metrics": photo_metrics.model_dump() if photo_metrics else None,
    }


class OpenRouterPromptWriter(PromptWriter):
    """PromptWriter that asks a cheap LLM, falling back to the deterministic writer."""

    def __init__(
        self,
        client: OpenRouterClient,
        *,
        model: str,
        fallback: PromptWriter | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._fallback = fallback if fallback is not None else DeterministicPromptWriter()

    async def compose(
        self,
        request: WriteRequest,
        *,
        photo_metrics: FrameMetrics | None = None,
        meter: BudgetMeter | None = None,
    ) -> WriteResult:
        return await self._call(
            _COMPOSE_SYSTEM,
            _request_view(request, photo_metrics),
            request=request,
            photo_metrics=photo_metrics,
            meter=meter,
        )

    async def revise(
        self,
        prev_body: str,
        feedback: WriterFeedback,
        *,
        request: WriteRequest,
        meter: BudgetMeter | None = None,
    ) -> WriteResult:
        view = _request_view(request, None)
        view["previous_body"] = prev_body
        view["previous_similarity"] = feedback.similarity
        view["previous_verdict"] = feedback.verdict.value if feedback.verdict else None
        return await self._call(
            _COMPOSE_SYSTEM + _REVISE_SUFFIX,
            view,
            request=request,
            photo_metrics=None,
            meter=meter,
            revising_from=prev_body,
        )

    async def _call(
        self,
        system: str,
        view: dict[str, Any],
        *,
        request: WriteRequest,
        photo_metrics: FrameMetrics | None,
        meter: BudgetMeter | None,
        revising_from: str | None = None,
    ) -> WriteResult:
        # Reserve before the paid call; a refused budget degrades to the
        # deterministic writer (or the prior body on a revise) rather than failing.
        reservation = None if meter is None else await meter.reserve(PaidCallKind.SLOT_FILL)
        if meter is not None and reservation is None:
            log.warning("prompt_writer_budget_exhausted")
            return await self._degrade(request, photo_metrics, revising_from)
        try:
            body = await self._client.chat_completion(self._payload(system, view))
            composed = _parse_body(body)
            usage = parse_usage(body)
            if reservation is None:
                return WriteResult(body=composed)
            cost = await reservation.settle(usage.cost if usage is not None else None)
            return WriteResult(body=composed, usage=usage, cost=cost)
        # Deliberately broad: whatever the LLM path did wrong, degrade. The
        # provider delivered no usable result, so the reservation is refunded.
        except Exception as exc:
            if reservation is not None:
                await reservation.cancel()
            log.warning("prompt_writer_llm_fallback", error=repr(exc))
            return await self._degrade(request, photo_metrics, revising_from)

    async def _degrade(
        self,
        request: WriteRequest,
        photo_metrics: FrameMetrics | None,
        revising_from: str | None,
    ) -> WriteResult:
        if revising_from is not None:
            return await self._fallback.revise(
                revising_from, WriterFeedback(None, None), request=request
            )
        return await self._fallback.compose(request, photo_metrics=photo_metrics)

    def _payload(self, system: str, view: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(view, ensure_ascii=False)},
            ],
            "temperature": 0.3,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "prompt_body",
                    "strict": True,
                    "schema": _response_schema(),
                },
            },
        }
