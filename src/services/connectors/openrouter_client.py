"""Shared OpenRouter HTTP transport — one place that speaks ``chat/completions``.

Both OpenRouter connectors (the image generator and the LLM slot filler) are
thin payload builders over this client. It owns the auth header, the timeout,
and the transient/permanent split: 429/5xx and network failures surface as
retryable (see :mod:`utils.retry` for why that is free), 4xx as
:class:`~utils.retry.UpstreamRequestError`, and a 2xx body is returned as-is —
interpreting it is the connector's job.

Tests inject an ``httpx.AsyncClient`` built on ``httpx.MockTransport`` so the
full request/response path runs without the network.
"""

from __future__ import annotations

from typing import Any

import httpx

from schemas import ProviderUsage
from utils.money import parse_usd
from utils.retry import TransientUpstreamError, UpstreamRequestError, with_transient_retry


def parse_usage(body: dict[str, Any]) -> ProviderUsage | None:
    """Pull the ``usage`` block (now always returned) off a completion body.

    ``cost`` is the dollars OpenRouter says it billed — the budget's source of
    truth; the token counts are observability. A missing/foreign ``usage`` shape
    collapses to ``None`` (the meter then settles on its reserved estimate), so
    accounting never crashes on a provider quirk.
    """
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    raw_cost = usage.get("cost")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return ProviderUsage(
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        cost=parse_usd(raw_cost) if isinstance(raw_cost, int | float | str) else None,
    )


class OpenRouterClient:
    """Authenticated POSTs to one OpenRouter endpoint, with transient retry."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        attempts: int = 4,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._attempts = attempts
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` to ``chat/completions`` and return the parsed body.

        Raises :class:`TransientUpstreamError` (after retries) when the upstream
        kept failing without a result, :class:`UpstreamRequestError` on 4xx.
        """

        async def _once() -> dict[str, Any]:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise TransientUpstreamError(
                    f"openrouter answered {response.status_code}",
                    status_code=response.status_code,
                )
            if response.status_code >= 400:
                # The body explains what we sent wrong; the prompt/images are
                # ours, so echoing the upstream's reason leaks nothing foreign.
                raise UpstreamRequestError(
                    f"openrouter rejected the request: {response.status_code} {response.text}",
                    status_code=response.status_code,
                )
            body: dict[str, Any] = response.json()
            return body

        return await with_transient_retry(_once, attempts=self._attempts)

    async def aclose(self) -> None:
        await self._client.aclose()
