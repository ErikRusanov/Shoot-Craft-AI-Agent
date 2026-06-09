"""In-memory object storage fake — a dict behind the ObjectStorage port.

No TTL, no network: ``put`` overwrites, ``get`` raises :class:`KeyError` for a
missing key, mirroring the real connectors' miss behavior.
"""

from __future__ import annotations


class InMemoryObjectStorage:
    """Blobs in a dict, addressed by key."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> str:
        self._data[key] = data
        return key

    async def get(self, key: str) -> bytes:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(key) from None
