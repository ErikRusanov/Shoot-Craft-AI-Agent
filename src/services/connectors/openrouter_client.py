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

from utils.retry import TransientUpstreamError, UpstreamRequestError, with_transient_retry


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
