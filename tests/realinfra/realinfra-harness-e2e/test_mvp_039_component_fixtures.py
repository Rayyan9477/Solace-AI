"""MVP-039 (IF-19): the real-infra component fixtures are real, isolated and honest.

Each fixture runs the deployed image (``IMAGE_REFS``) in a container of its own. A missing Docker daemon
fails the run when ``SOLACE_REQUIRE_REALINFRA=1`` instead of skipping it.
"""
from __future__ import annotations

import asyncio
import json
import smtplib
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
import redis
import redis.asyncio
import redis.exceptions
import yaml

from solace_events.schemas import SessionStartedEvent
from solace_testing import realinfra


async def test_mvp_039_postgres_is_migrated_to_head_and_app_role_is_constrained(pg_migrated):
    from solace_infrastructure.database.schema_check import SCHEMA_HEAD
    (v,) = await pg_migrated.superuser_fetch("SELECT version_num FROM alembic_version")
    assert v["version_num"] == SCHEMA_HEAD
    async with pg_migrated.app_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT current_user AS u, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
    assert (row["u"], row["rolsuper"], row["rolbypassrls"]) == ("solace_app", False, False)
    assert "solace_app" not in repr(pg_migrated) or "***" in repr(pg_migrated)

async def test_mvp_039_app_pool_binds_jsonb_as_objects(pg_migrated):
    async with pg_migrated.app_pool.acquire() as conn:
        assert await conn.fetchval("SELECT jsonb_typeof($1::jsonb)", {"a": 1}) == "object"

async def test_mvp_039_create_database_stops_at_the_requested_revision(pg_migrated):
    db = await pg_migrated.create_database(f"h039_{uuid4().hex[:8]}", migrate_to="009_platform_roles_rls")
    (v,) = await db.superuser_fetch("SELECT version_num FROM alembic_version")
    assert v["version_num"] == "009_platform_roles_rls"

async def test_mvp_039_redis_requires_the_password(redis_url):
    assert await redis.asyncio.from_url(redis_url).ping() is True
    with pytest.raises(redis.exceptions.AuthenticationError):
        await redis.asyncio.from_url(redis_url.replace(f":{urlsplit(redis_url).password}@", "@")).ping()

async def test_mvp_039_kafka_round_trip_and_committed_offset(kafka_bootstrap):
    topic, ev = f"test.h039.{uuid4().hex[:8]}", SessionStartedEvent(user_id=uuid4(), session_number=1)
    await kafka_bootstrap.produce(topic, ev)
    got = await kafka_bootstrap.consume_until(topic, lambda m: m["metadata"]["event_id"] == str(ev.metadata.event_id), timeout=30, group_id="g-h039")
    assert len(got) == 1 and await kafka_bootstrap.committed_offset("g-h039", topic) == 1

def test_mvp_039_smtp_sink_captures_mail(smtp_sink):
    with smtplib.SMTP(smtp_sink.smtp_host, smtp_sink.smtp_port) as s:
        s.sendmail("noreply@test.invalid", ["clin-h039@test.invalid"], "Subject: [Solace CRISIS HIGH] check\r\n\r\nbody")
    assert smtp_sink.wait_for("clin-h039@test.invalid", "[Solace CRISIS", timeout=10).subject == "[Solace CRISIS HIGH] check"

def test_mvp_039_fake_llm_container_answers(fake_llm):
    r = httpx.post(f"{fake_llm.base_url}/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["object"] == "chat.completion"

def test_mvp_039_missing_docker_fails_when_required(monkeypatch):
    monkeypatch.setenv("SOLACE_REQUIRE_REALINFRA", "1"); monkeypatch.setattr(realinfra, "_docker_ping", lambda: False)
    with pytest.raises(pytest.fail.Exception):
        realinfra.require_docker()


# --- WS-05.2 fix round 1 (review R1-R5): no fixture lets a test pass without proving anything ---------------

FAKE_REPLY = "Thank you for telling me. I'm here with you."
_ISOLATION_ORDER: list[str] = []


def _chat(fake_llm, text: str) -> str:
    r = httpx.post(f"{fake_llm.base_url}/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": text}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def test_mvp_039_isolation_a_leaves_mail_llm_and_redis_state(smtp_sink, fake_llm, redis_url):
    with smtplib.SMTP(smtp_sink.smtp_host, smtp_sink.smtp_port) as s:
        s.sendmail("noreply@test.invalid", ["clin-iso@test.invalid"], "Subject: [Solace CRISIS HIGH] from test A\r\n\r\nbody")
    assert smtp_sink.wait_for("clin-iso@test.invalid", "[Solace CRISIS", timeout=10)
    fake_llm.scenario("crisis_intent")
    assert "crisis_disclosure" in _chat(fake_llm, "anything")
    client = redis.Redis.from_url(redis_url)
    client.set("h039:iso", "from test A")
    client.close()
    _ISOLATION_ORDER.append("a")


def test_mvp_039_isolation_b_starts_from_clean_fixtures(smtp_sink, fake_llm, redis_url):
    assert _ISOLATION_ORDER == ["a"], "must run right after test_mvp_039_isolation_a_leaves_mail_llm_and_redis_state"
    assert smtp_sink.messages() == []
    with pytest.raises(TimeoutError):
        smtp_sink.wait_for("clin-iso@test.invalid", "[Solace CRISIS", timeout=1)
    assert fake_llm.requests() == []
    assert _chat(fake_llm, "I had a nice day") == FAKE_REPLY
    client = redis.Redis.from_url(redis_url)
    try:
        assert client.get("h039:iso") is None
    finally:
        client.close()


def _broker_status(kafka) -> str:
    wrapped = kafka._wrapped()
    wrapped.reload()
    return wrapped.status


async def test_mvp_039_a_broker_left_paused_fails_the_test_and_is_resumed(kafka_bootstrap):
    kafka_bootstrap.pause()
    try:
        with pytest.raises(pytest.fail.Exception, match="left the kafka broker paused"):
            realinfra._check_kafka_running(kafka_bootstrap)
        assert _broker_status(kafka_bootstrap) == "running"
    finally:
        if _broker_status(kafka_bootstrap) == "paused":
            kafka_bootstrap.unpause()
    await kafka_bootstrap.produce(f"test.h039.{uuid4().hex[:8]}", {"after": "unpause"})


async def test_mvp_039_consume_until_counts_a_duplicate_published_after_the_first_match(kafka_bootstrap):
    topic, ev = f"test.h039.{uuid4().hex[:8]}", SessionStartedEvent(user_id=uuid4(), session_number=1)
    first_match = asyncio.Event()

    def is_ev(m):
        hit = m["metadata"]["event_id"] == str(ev.metadata.event_id)
        if hit:
            first_match.set()
        return hit

    async def late_duplicate():
        await first_match.wait()
        await asyncio.sleep(1.0)
        await kafka_bootstrap.produce(topic, ev)

    await kafka_bootstrap.produce(topic, ev)
    late = asyncio.create_task(late_duplicate())
    try:
        got = await kafka_bootstrap.consume_until(topic, is_ev, timeout=30, settle=3.0)
    except BaseException:
        late.cancel()
        raise
    await asyncio.wait_for(late, 30)
    assert len(got) == 2


async def test_mvp_039_consume_until_raises_instead_of_an_unread_empty_result(kafka_bootstrap):
    # timeout=0 leaves no time to join the group and read: the matching message is there, unread. (A tiny positive
    # timeout is not deterministic: a warm coordinator can assign and fetch within the first poll.)
    topic, ev = f"test.h039.{uuid4().hex[:8]}", SessionStartedEvent(user_id=uuid4(), session_number=1)
    await kafka_bootstrap.produce(topic, ev)
    with pytest.raises(TimeoutError, match="did not read"):
        await kafka_bootstrap.consume_until(topic, lambda m: True, timeout=0, group_id=f"g-h039-{uuid4().hex[:8]}")


async def test_mvp_039_consume_until_returns_empty_only_after_reading_the_whole_topic(kafka_bootstrap):
    topic = f"test.h039.{uuid4().hex[:8]}"
    await kafka_bootstrap.produce(topic, {"benign": 1})
    assert await kafka_bootstrap.consume_until(topic, lambda m: "crisis" in m, timeout=3) == []


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"mode": "reply"}, "needs content"),
    ({"mode": "reply", "content": 123}, "needs content"),
    ({"mode": "ok", "content": "ignored text"}, "content applies only"),
    ({"mode": "echo", "match": 123}, "match must be"),
    ({"mode": "echo", "match": ""}, "match must be"),
])
def test_mvp_039_fake_llm_rejects_malformed_scenarios(fake_llm, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fake_llm.scenario(**kwargs)
    assert _chat(fake_llm, "hello") == FAKE_REPLY  # the server never received it


# --- WS-05.2 fix round 2 (re-review N1): the settle phase is bounded from the first match and says why it stops ---


class _ScriptedConsumer:
    """Stands in for AIOKafkaConsumer: one partition whose records reach the broker at scripted times (seconds
    after start). ``end_extra`` adds records the broker reports but never delivers (a stuck fetch)."""

    def __init__(self, script: list[tuple[float, dict]], end_extra: int = 0) -> None:
        self.script, self.end_extra, self.delivered, self.t0 = script, end_extra, 0, 0.0

    def __call__(self, topic: str, **kwargs):  # replaces the class: AIOKafkaConsumer(topic, ...)
        return self

    def _due(self) -> int:
        now = time.monotonic() - self.t0
        return sum(1 for at, _ in self.script if at <= now)

    async def start(self) -> None:
        self.t0 = time.monotonic()

    async def stop(self) -> None:
        pass

    def assignment(self) -> set[str]:
        return {"p0"}

    async def end_offsets(self, partitions) -> dict[str, int]:
        return {"p0": self._due() + self.end_extra}

    async def position(self, tp) -> int:
        return self.delivered

    async def commit(self) -> None:
        pass

    async def getmany(self, timeout_ms: int) -> dict:
        until = time.monotonic() + timeout_ms / 1000
        while self._due() <= self.delivered and time.monotonic() < until:
            await asyncio.sleep(0.01)
        due = self._due()
        records = [SimpleNamespace(value=json.dumps(m).encode()) for _, m in self.script[self.delivered:due]]
        self.delivered = due
        return {"p0": records} if records else {}


def _scripted(monkeypatch, script, end_extra=0) -> realinfra.KafkaHandle:
    import aiokafka
    monkeypatch.setattr(aiokafka, "AIOKafkaConsumer", _ScriptedConsumer(script, end_extra))
    return realinfra.KafkaHandle(bootstrap="scripted:0")


async def test_mvp_039_a_duplicate_after_the_deadline_inside_the_first_window_is_counted(monkeypatch):
    # First match at 0.5 s, its duplicate at 0.8 s: after the 0.6 s timeout, inside the first match's 0.6 s window.
    kafka = _scripted(monkeypatch, [(0.5, {"id": 1}), (0.8, {"id": 1})])
    got = await kafka.consume_until("t", lambda m: m["id"] == 1, timeout=0.6, settle=0.6)
    assert len(got) == 2


async def test_mvp_039_a_stream_still_matching_at_the_bound_raises_still_arriving(monkeypatch):
    kafka = _scripted(monkeypatch, [(0.2 * i, {"id": 1}) for i in range(1, 40)])
    with pytest.raises(TimeoutError, match="still arriving"):
        await kafka.consume_until("t", lambda m: m["id"] == 1, timeout=0.6, settle=0.6)


async def test_mvp_039_a_match_whose_end_offsets_are_never_reached_raises(monkeypatch):
    kafka = _scripted(monkeypatch, [(0.1, {"id": 1})], end_extra=4)
    with pytest.raises(TimeoutError, match="did not read to the end offsets"):
        await kafka.consume_until("t", lambda m: m["id"] == 1, timeout=0.3, settle=0.3)


def test_mvp_039_image_refs_check_only_the_fixture_images(tmp_path):
    compose = tmp_path / "docker-compose.yml"
    services = {"postgres": {"image": "postgres:16-alpine"}, "redis": {"image": "redis:7-alpine"},
                "kafka": {"image": "confluentinc/cp-kafka:7.5.0"},
                "user-service": {"image": "ghcr.io/solace/user-service:${TAG:-dev}"}}
    compose.write_text(yaml.safe_dump({"services": services}), encoding="utf-8")
    refs = realinfra._image_refs([compose])
    assert refs["postgres"] == "postgres:16-alpine" and "user-service" not in refs
    services["postgres"]["image"] = "postgres:${PG_TAG:-16-alpine}"
    compose.write_text(yaml.safe_dump({"services": services}), encoding="utf-8")
    with pytest.raises(ValueError, match="postgres"):
        realinfra._image_refs([compose])
