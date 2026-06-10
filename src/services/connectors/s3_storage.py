"""S3-backed :class:`~protocols.object_storage.ObjectStorage` (aioboto3).

Works against AWS S3 and any S3-compatible endpoint (MinIO in tests,
``s3_endpoint_url`` in config). The underlying client is an async context
manager, so it is opened lazily on first use and held until :meth:`aclose` —
one connection pool per store, not per call. A missing key surfaces as
:class:`KeyError`, the contract's miss behavior.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

import aioboto3
from botocore.exceptions import ClientError


class S3ObjectStorage:
    """Blobs as objects in one S3 bucket."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._region = region
        self._session = aioboto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._stack = AsyncExitStack()
        self._client: Any = None

    async def _s3(self) -> Any:
        if self._client is None:
            self._client = await self._stack.enter_async_context(
                self._session.client("s3", endpoint_url=self._endpoint_url)
            )
        return self._client

    async def put(self, key: str, data: bytes) -> str:
        s3 = await self._s3()
        await s3.put_object(Bucket=self._bucket, Key=key, Body=data)
        return key

    async def get(self, key: str) -> bytes:
        s3 = await self._s3()
        try:
            response = await s3.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise KeyError(key) from None
            raise
        async with response["Body"] as body:
            data = await body.read()
        return bytes(data)

    async def aclose(self) -> None:
        await self._stack.aclose()
        self._client = None
