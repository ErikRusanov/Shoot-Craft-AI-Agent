"""Retry policy for outbound HTTP — transient failures only, with jitter.

The split that matters for money: a **transient** failure (network error,
timeout, 429/5xx) means the upstream never delivered a result, so retrying is
free — nothing was paid for. A successful response is *never* retried here, and
a 4xx is a contract bug on our side, not weather — both fail straight through
to the caller. Budget accounting therefore only ever sees calls that actually
returned a result.

Both error classes live here, next to the policy that interprets them, so a
connector imports the transient/permanent split from one place.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

log = structlog.get_logger(__name__)


class TransientUpstreamError(Exception):
    """The upstream failed without delivering a result — safe to retry.

    Raised for 429 and 5xx responses; network-level failures keep their native
    ``httpx.TransportError`` (which the policy treats the same way).
    """

    def __init__(self, detail: str, *, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code


class UpstreamRequestError(Exception):
    """The upstream rejected the request (4xx) — retrying verbatim cannot help."""

    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.status_code = status_code


def _is_transient(exc: BaseException) -> bool:
    # TransportError covers timeouts, connection drops and protocol errors —
    # all cases where no response (and hence no charge) materialized.
    return isinstance(exc, TransientUpstreamError | httpx.TransportError)


def _log_retry(retry_state: RetryCallState) -> None:
    # No payloads, no PII — just enough to see flapping upstreams in ops.
    log.warning(
        "transient_upstream_retry",
        attempt=retry_state.attempt_number,
        error=repr(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


async def with_transient_retry[T](fn: Callable[[], Awaitable[T]], *, attempts: int = 4) -> T:
    """Run ``fn``, retrying only transient failures; everything else propagates.

    Exponential backoff with full jitter so simultaneous sessions don't
    re-stampede a recovering upstream. After ``attempts`` tries the original
    exception is re-raised (``reraise=True``), not tenacity's wrapper.
    """
    retryer = AsyncRetrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(attempts),
        wait=wait_random_exponential(multiplier=0.5, max=8.0),
        before_sleep=_log_retry,
        reraise=True,
    )
    return await retryer(fn)
