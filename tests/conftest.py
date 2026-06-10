"""Shared fixtures: every port fixture is parametrized over its implementations.

Contract and reliability tests request ``store`` / ``bus`` / ``storage`` and run
once per backend. In-memory backends run anywhere; Redis- and MinIO-backed ones
spin real servers via testcontainers and are skipped when no Docker daemon is
available (CI always has one, so nothing silently skips there).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from uuid import uuid4

import aioboto3
import pytest
from redis.asyncio import Redis
from testcontainers.minio import MinioContainer
from testcontainers.redis import RedisContainer

from protocols import EventBus, ObjectStorage, StateStore
from services.connectors import (
    InMemoryEventBus,
    InMemoryStateStore,
    LocalObjectStorage,
    RedisEventBus,
    RedisStateStore,
    S3ObjectStorage,
)
from tests.fakes import InMemoryObjectStorage

REDIS_IMAGE = "redis:8-alpine"
S3_REGION = "us-east-1"


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """One Redis server for the whole run; tests isolate by flushing per test."""
    container = RedisContainer(REDIS_IMAGE)
    try:
        container.start()
    except Exception:
        pytest.skip("Docker is not available — skipping Redis-backed tests")
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(container.port)
        yield f"redis://{host}:{port}/0"
    finally:
        container.stop()


@pytest.fixture(scope="session")
def minio_config() -> Iterator[dict[str, str]]:
    """One MinIO server for the whole run; tests isolate by bucket."""
    container = MinioContainer()
    try:
        container.start()
    except Exception:
        pytest.skip("Docker is not available — skipping S3-backed tests")
    try:
        cfg = container.get_config()
        yield {
            "endpoint_url": f"http://{cfg['endpoint']}",
            "access_key": cfg["access_key"],
            "secret_key": cfg["secret_key"],
        }
    finally:
        container.stop()


@pytest.fixture(params=["memory", "redis"])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[StateStore]:
    if request.param == "memory":
        yield InMemoryStateStore()
        return
    client = Redis.from_url(request.getfixturevalue("redis_url"))
    await client.flushdb()
    yield RedisStateStore(client)
    await client.aclose()


@pytest.fixture(params=["memory", "redis"])
async def bus(request: pytest.FixtureRequest) -> AsyncIterator[EventBus]:
    if request.param == "memory":
        yield InMemoryEventBus()
        return
    client = Redis.from_url(request.getfixturevalue("redis_url"))
    await client.flushdb()
    yield RedisEventBus(client)
    await client.aclose()


@pytest.fixture(params=["memory", "local", "s3"])
async def storage(request: pytest.FixtureRequest) -> AsyncIterator[ObjectStorage]:
    if request.param == "memory":
        yield InMemoryObjectStorage()
        return
    if request.param == "local":
        yield LocalObjectStorage(request.getfixturevalue("tmp_path"))
        return

    cfg = request.getfixturevalue("minio_config")
    bucket = f"test-{uuid4().hex[:12]}"
    session = aioboto3.Session(
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        region_name=S3_REGION,
    )
    async with session.client("s3", endpoint_url=cfg["endpoint_url"]) as s3:
        await s3.create_bucket(Bucket=bucket)
    s3_storage = S3ObjectStorage(
        bucket=bucket,
        endpoint_url=cfg["endpoint_url"],
        region=S3_REGION,
        access_key=cfg["access_key"],
        secret_key=cfg["secret_key"],
    )
    yield s3_storage
    await s3_storage.aclose()
