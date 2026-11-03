"""REV-17: memory-service GDPR right-to-erasure wiring.

A user's right to erasure (GDPR Art. 17) spans every store the memory service
owns: the Postgres ``memory_records`` / ``session_summaries`` / ``user_facts`` /
``therapeutic_events`` tables, the Weaviate vector collections, and the Redis
cache. These tests encode the acceptance criteria:

  (a) ``RedisCache.delete_user_cache`` purges EVERY user key shape
      (``working:{uid}:{sid}``, ``context:{uid}:{ck}``, ``user_session:{uid}``,
      ``session_count:{uid}`` and the uid-less ``session:{sid}`` blob) and
      touches NO other user's keys (nor the shared content-addressed embedding
      cache);
  (b) the factory registers EXACTLY the three memory stores;
  (c) ``erase`` fans out to all three stores and returns their counts;
  (d) if any store deleter raises, ``erase`` raises ``ErasureIncomplete``
      (fail-loud) after still attempting the others; and the domain
      ``MemoryService.delete_user_data`` drives everything through the
      orchestrator while preserving its ``None`` (HTTP 204) response contract.
"""
from __future__ import annotations

import fnmatch
import json
from uuid import uuid4

import pytest

from solace_infrastructure.gdpr import ErasureIncomplete, UserDataErasure

from services.memory_service.src.domain.erasure import (
    MEMORY_POSTGRES_STORE,
    MEMORY_REDIS_STORE,
    MEMORY_WEAVIATE_STORE,
    build_user_data_erasure,
)
from services.memory_service.src.domain.models import MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.infrastructure.redis_cache import (
    RedisCache,
    RedisSettings,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeRedisClient:
    """Minimal in-memory stand-in for a redis.asyncio client.

    Implements only what ``delete_user_cache`` uses: ``scan`` (glob match, single
    shot), ``get`` and ``delete``. ``fnmatchcase`` mirrors Redis ``*`` semantics
    (``*`` spans ``:`` just like Redis, and matching is case-sensitive).
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def scan(self, cursor, match=None, count=100):  # noqa: ANN001
        if match is None:
            keys = list(self.store)
        else:
            keys = [k for k in self.store if fnmatch.fnmatchcase(k, match)]
        return 0, keys

    async def get(self, key):  # noqa: ANN001
        return self.store.get(key)

    async def delete(self, *keys):
        removed = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                removed += 1
        return removed


class _FakePostgres:
    """Fake memory Postgres repo: delete_user_data returns the 4-tuple contract."""

    def __init__(self) -> None:
        self.deleted: list = []

    async def delete_user_data(self, user_id):  # noqa: ANN001
        self.deleted.append(user_id)
        return (2, 1, 3, 0)  # records, summaries, facts, events


class _FakeWeaviate:
    def __init__(self) -> None:
        self.deleted: list = []

    async def delete_user_data(self, user_id):  # noqa: ANN001
        self.deleted.append(user_id)
        return 3  # collections purged


class _FakeRedisCache:
    def __init__(self) -> None:
        self.deleted: list = []

    async def delete_user_cache(self, user_id):  # noqa: ANN001
        self.deleted.append(user_id)
        return 5  # keys purged


class _RaisingPostgres:
    async def delete_user_data(self, user_id):  # noqa: ANN001
        raise RuntimeError("db pool exhausted")


# ---------------------------------------------------------------------------
# (a) Redis purge covers every user key shape, spares everyone else
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_user_cache_purges_every_user_key_shape_and_spares_others() -> None:
    cache = RedisCache(RedisSettings(key_prefix="solace:memory:"))
    fake = FakeRedisClient()
    cache._client = fake
    cache._initialized = True

    target = uuid4()
    other = uuid4()
    t_sid = uuid4()
    t_orphan_sid = uuid4()  # a prior-session blob with no user_session pointer
    o_sid = uuid4()

    def sess_blob(uid, sid) -> str:
        return json.dumps(
            {"session_id": str(sid), "user_id": str(uid), "message_count": 1}
        )

    # Target user: one key of EVERY shape, plus an orphaned session blob.
    fake.store[cache._key("working", str(target), str(t_sid))] = json.dumps(
        {"user_id": str(target)}
    )
    fake.store[cache._key("context", str(target), "ctx-key")] = "assembled-context"
    fake.store[cache._key("user_session", str(target))] = str(t_sid)
    fake.store[cache._key("session_count", str(target))] = "3"
    fake.store[cache._key("session", str(t_sid))] = sess_blob(target, t_sid)
    fake.store[cache._key("session", str(t_orphan_sid))] = sess_blob(target, t_orphan_sid)

    # Other user: must survive untouched.
    survivors = {
        cache._key("working", str(other), str(o_sid)): json.dumps({"user_id": str(other)}),
        cache._key("context", str(other), "ck"): "o-ctx",
        cache._key("user_session", str(other)): str(o_sid),
        cache._key("session_count", str(other)): "1",
        cache._key("session", str(o_sid)): sess_blob(other, o_sid),
        # Content-addressed embedding cache is shared (keyed by hash, not user).
        cache._key("embedding", "deadbeef"): "[0.1,0.2]",
    }
    fake.store.update(survivors)

    deleted = await cache.delete_user_cache(target)

    # Every target key shape purged: middle-uid (working/context), trailing-uid
    # (user_session/session_count) and the two uid-less session blobs.
    assert not any(str(target) in k for k in fake.store), fake.store
    assert cache._key("session", str(t_sid)) not in fake.store
    assert cache._key("session", str(t_orphan_sid)) not in fake.store
    assert deleted == 6

    # Nobody else's keys were touched (proves no over-reach), embedding survives.
    assert set(fake.store) == set(survivors)
    assert cache._key("session", str(o_sid)) in fake.store
    assert cache._key("embedding", "deadbeef") in fake.store


@pytest.mark.asyncio
async def test_delete_user_cache_noop_when_uninitialized() -> None:
    cache = RedisCache(RedisSettings())
    # No client attached; guarded path returns 0 without raising.
    assert await cache.delete_user_cache(uuid4()) == 0


# ---------------------------------------------------------------------------
# (b) factory registers exactly the three memory stores
# ---------------------------------------------------------------------------
def test_factory_registers_exactly_the_three_memory_stores() -> None:
    erasure = build_user_data_erasure(
        postgres_repo=_FakePostgres(),
        weaviate_repo=_FakeWeaviate(),
        redis_cache=_FakeRedisCache(),
    )
    assert isinstance(erasure, UserDataErasure)
    assert erasure.store_names == (
        MEMORY_POSTGRES_STORE,
        MEMORY_WEAVIATE_STORE,
        MEMORY_REDIS_STORE,
    )
    assert (MEMORY_POSTGRES_STORE, MEMORY_WEAVIATE_STORE, MEMORY_REDIS_STORE) == (
        "memory_postgres",
        "memory_weaviate",
        "memory_redis",
    )


def test_factory_registers_only_provided_stores() -> None:
    erasure = build_user_data_erasure(redis_cache=_FakeRedisCache())
    assert erasure.store_names == (MEMORY_REDIS_STORE,)


# ---------------------------------------------------------------------------
# (c) erase() fans out to all three stores and returns their counts
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_erase_fans_out_to_all_three_stores_and_returns_counts() -> None:
    pg, wv, rd = _FakePostgres(), _FakeWeaviate(), _FakeRedisCache()
    erasure = build_user_data_erasure(postgres_repo=pg, weaviate_repo=wv, redis_cache=rd)
    uid = uuid4()

    report = await erasure.erase(uid)

    assert report.complete is True
    assert report.failed == {}
    assert report.deleted[MEMORY_POSTGRES_STORE] == 6  # 2 + 1 + 3 + 0
    assert report.deleted[MEMORY_WEAVIATE_STORE] == 3
    assert report.deleted[MEMORY_REDIS_STORE] == 5
    assert report.total_deleted == 14
    # Every store's deleter was invoked exactly once with the user id.
    assert pg.deleted == [uid]
    assert wv.deleted == [uid]
    assert rd.deleted == [uid]


# ---------------------------------------------------------------------------
# (d) fail-loud when a store raises; domain method routes + preserves contract
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_erase_fails_loud_when_one_store_raises_but_still_attempts_others() -> None:
    pg = _RaisingPostgres()
    wv, rd = _FakeWeaviate(), _FakeRedisCache()
    erasure = build_user_data_erasure(postgres_repo=pg, weaviate_repo=wv, redis_cache=rd)
    uid = uuid4()

    with pytest.raises(ErasureIncomplete) as exc_info:
        await erasure.erase(uid)

    report = exc_info.value.report
    assert report.complete is False
    assert MEMORY_POSTGRES_STORE in report.failed
    assert report.audit_emitted is False
    # Fail-loud, NOT fail-fast: the other stores were still attempted.
    assert report.deleted.get(MEMORY_WEAVIATE_STORE) == 3
    assert report.deleted.get(MEMORY_REDIS_STORE) == 5
    assert wv.deleted == [uid] and rd.deleted == [uid]


@pytest.mark.asyncio
async def test_domain_delete_user_data_drives_all_stores_and_preserves_contract() -> None:
    pg, wv, rd = _FakePostgres(), _FakeWeaviate(), _FakeRedisCache()
    service = MemoryService(
        settings=MemoryServiceSettings(),
        postgres_repo=pg,
        weaviate_repo=wv,
        redis_cache=rd,
    )
    uid = uuid4()
    service._tier_3_session[uid] = ["seed"]
    service._user_session_counts[uid] = 2

    result = await service.delete_user_data(user_id=uid)

    assert result is None  # existing HTTP 204 contract preserved
    assert pg.deleted == [uid]
    assert wv.deleted == [uid]
    assert rd.deleted == [uid]  # Redis now wired in (was never called before)
    # In-memory tier caches cleared after the durable stores were purged.
    assert uid not in service._tier_3_session
    assert uid not in service._user_session_counts


@pytest.mark.asyncio
async def test_domain_delete_user_data_propagates_incomplete_erasure() -> None:
    service = MemoryService(
        settings=MemoryServiceSettings(),
        postgres_repo=_RaisingPostgres(),
        weaviate_repo=_FakeWeaviate(),
        redis_cache=_FakeRedisCache(),
    )
    uid = uuid4()
    service._tier_3_session[uid] = ["seed"]

    with pytest.raises(ErasureIncomplete):
        await service.delete_user_data(user_id=uid)

    # A failed durable erase must NOT be reported complete: in-memory caches are
    # left intact (cleared only AFTER the durable stores are fully purged).
    assert uid in service._tier_3_session


# ---------------------------------------------------------------------------
# (d') Real store deleters fail LOUD on the erasure path (no false "complete").
#      The fakes above return ints; these exercise the real WeaviateRepository /
#      RedisCache so a swallowed error can never masquerade as a successful erase.
# ---------------------------------------------------------------------------
class _RaisingWeaviateColl:
    """coll.data.delete_many(...) raises — a Weaviate outage mid-erase."""

    def __init__(self) -> None:
        self.data = self

    def delete_many(self, where=None):  # noqa: ANN001 - runs via asyncio.to_thread
        raise RuntimeError("weaviate unavailable")


class _RaisingWeaviateClient:
    class _Collections:
        def get(self, name):  # noqa: ANN001
            return _RaisingWeaviateColl()

    def __init__(self) -> None:
        self.collections = _RaisingWeaviateClient._Collections()


@pytest.mark.asyncio
async def test_weaviate_delete_user_data_raises_when_uninitialized() -> None:
    from services.memory_service.src.infrastructure.weaviate_repo import (
        WeaviateRepository,
    )

    repo = WeaviateRepository()  # _initialized is False, no client
    # Registered-but-unreachable unique-vector store must NOT report a silent 0.
    with pytest.raises(RuntimeError):
        await repo.delete_user_data(uuid4())


@pytest.mark.asyncio
async def test_weaviate_delete_user_data_raises_on_delete_failure() -> None:
    from services.memory_service.src.infrastructure.weaviate_repo import (
        WeaviateRepository,
    )

    repo = WeaviateRepository()
    repo._client = _RaisingWeaviateClient()
    repo._initialized = True
    # Any per-collection delete_many failure is collected and re-raised so the
    # orchestrator records memory_weaviate as failed (ErasureIncomplete).
    with pytest.raises(RuntimeError):
        await repo.delete_user_data(uuid4())


class _RaisingDeleteRedisClient(FakeRedisClient):
    """scan/get behave normally, but delete() raises mid-purge."""

    async def delete(self, *keys):
        raise RuntimeError("redis connection reset")


@pytest.mark.asyncio
async def test_redis_delete_user_cache_reraises_operational_failure() -> None:
    cache = RedisCache(RedisSettings(key_prefix="solace:memory:"))
    fake = _RaisingDeleteRedisClient()
    cache._client = fake
    cache._initialized = True
    uid = uuid4()
    fake.store[cache._key("working", str(uid), str(uuid4()))] = json.dumps(
        {"user_id": str(uid)}
    )
    # An operational Redis failure mid-purge is fail-loud (raised), never a
    # swallowed "0 keys removed, complete" attestation.
    with pytest.raises(RuntimeError):
        await cache.delete_user_cache(uid)
