"""Port: object storage — opaque bytes in, bytes out, by key.

Holds the input photos and generated frames; :class:`~schemas.state.FaceProfile`
and :class:`~schemas.state.Iteration` reference them only by key (``photo_ref`` /
``result_ref``), never inline. S3 in prod, a local directory in dev — both behind
this port so the rest of the code only ever sees a string key.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectStorage(Protocol):
    """Store and fetch opaque blobs addressed by an externally chosen key."""

    async def put(self, key: str, data: bytes) -> str:
        """Store ``data`` at ``key`` (overwriting) and echo the key back."""
        ...

    async def get(self, key: str) -> bytes:
        """Return the bytes at ``key``; raise :class:`KeyError` if there is none."""
        ...
