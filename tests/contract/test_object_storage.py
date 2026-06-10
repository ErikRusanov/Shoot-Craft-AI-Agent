"""Contract: :class:`~protocols.object_storage.ObjectStorage`.

Pins the blob round-trip, the key echo, last-write-wins, and the miss behavior
(:class:`KeyError`) every backend must share. The ``storage`` fixture
(tests/conftest.py) parametrizes every test over all implementations:
in-memory, local directory, and S3 (MinIO).
"""

from __future__ import annotations

import pytest

from protocols import ObjectStorage


async def test_is_object_storage(storage: ObjectStorage) -> None:
    assert isinstance(storage, ObjectStorage)


async def test_put_get_roundtrip(storage: ObjectStorage) -> None:
    returned = await storage.put("results/img-1.png", b"\x89PNG-bytes")
    assert returned == "results/img-1.png"
    assert await storage.get("results/img-1.png") == b"\x89PNG-bytes"


async def test_get_missing_raises_keyerror(storage: ObjectStorage) -> None:
    with pytest.raises(KeyError):
        await storage.get("nope")


async def test_put_overwrites(storage: ObjectStorage) -> None:
    await storage.put("k", b"first")
    await storage.put("k", b"second")
    assert await storage.get("k") == b"second"
