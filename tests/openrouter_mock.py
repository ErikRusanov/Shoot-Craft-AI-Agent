"""Mocked-httpx harness for the OpenRouter connectors.

Everything OpenRouter-shaped that tests need in one place: canned
``chat/completions`` bodies, a scripted transport that records every request
and can fail on cue (an entry may be an exception to raise instead of a
response), and a factory wiring it all into a real :class:`OpenRouterClient` —
so tests exercise the genuine request/retry path with zero network.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import httpx
from PIL import Image

from services.connectors import OpenRouterClient

BASE_URL = "https://openrouter.test/api/v1"
API_KEY = "test-key"


def _pixel_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (127, 127, 127)).save(buf, format="PNG")
    return buf.getvalue()


PIXEL_PNG = _pixel_png()


def image_completion_body(
    *, image: bytes = PIXEL_PNG, request_id: str = "gen-test", mime: str = "image/png"
) -> dict[str, Any]:
    """A successful generation response in OpenRouter's documented shape."""
    data_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
    return {
        "id": request_id,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "images": [{"type": "image_url", "image_url": {"url": data_url}}],
                }
            }
        ],
    }


def text_completion_body(content: str, *, request_id: str = "gen-test") -> dict[str, Any]:
    """A text-only completion (the slot filler's shape; also a no-image 200)."""
    return {
        "id": request_id,
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


class ScriptedTransport:
    """Returns the scripted entries in order; the last one repeats forever.

    An entry that is an exception is raised instead of returned, so transport
    failures (timeouts, connection drops) are scripted the same way as bad
    status codes. Every incoming request is recorded for shape assertions.
    """

    def __init__(self, *script: httpx.Response | Exception) -> None:
        self.requests: list[httpx.Request] = []
        self._script = list(script)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        entry = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(entry, Exception):
            raise entry
        return entry


def scripted_client(
    *script: httpx.Response | Exception, attempts: int = 4
) -> tuple[OpenRouterClient, ScriptedTransport]:
    """A real OpenRouterClient over a scripted in-memory transport."""
    transport = ScriptedTransport(*script)
    client = OpenRouterClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout_seconds=5.0,
        attempts=attempts,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    return client, transport
