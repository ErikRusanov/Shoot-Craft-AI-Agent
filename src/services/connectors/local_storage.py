"""Local-directory :class:`~protocols.object_storage.ObjectStorage` (dev fallback).

Keys map to paths under a root directory; slashes in keys become
subdirectories. File I/O runs in ``asyncio.to_thread`` to keep the event loop
unblocked. Dev-only by design — no TTL, no replication — but the contract
(round-trip, overwrite, ``KeyError`` on miss) matches S3 exactly.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


class LocalObjectStorage:
    """Blobs as files under a root directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        # A key is storage-internal, never user input — but refuse traversal
        # outside the root anyway rather than scribble over the filesystem.
        if not path.is_relative_to(self._root):
            raise ValueError(f"object key escapes storage root: {key!r}")
        return path

    def _write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a concurrent get never sees a torn blob.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    async def put(self, key: str, data: bytes) -> str:
        await asyncio.to_thread(self._write, key, data)
        return key

    async def get(self, key: str) -> bytes:
        try:
            return await asyncio.to_thread(self._path(key).read_bytes)
        except FileNotFoundError:
            raise KeyError(key) from None
