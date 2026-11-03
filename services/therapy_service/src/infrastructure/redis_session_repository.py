"""
Solace-AI Therapy Service - Redis Session Repository (REV-16 groundwork).

A Redis-backed implementation of the therapy ``SessionRepository`` port
(``domain/session_manager.py``): live session state (phase, agenda, safety
flags, engagement, nested technique/homework DTOs) serialized to JSON, keyed by
session id with a TTL, so it CAN survive a process restart and be shared across
workers.

✅ WIRED (REV-16 / Stage 2 B1, 2026-07-30). ``TherapyOrchestrator`` now write-throughs
session state to this store on every mutation and reads-through on a cache miss
(``domain/service.py`` ``_persist_session`` / ``_load_session``). Concurrency safety:
  1. optimistic concurrency via a ``version`` field + ``try_save_session`` (version
     check under Redis ``WATCH`` → ``MULTI/EXEC``), and ``save_session_cas_merge``
     which, on a conflict, reloads the committed state, MERGES the load-bearing
     safety fields (:func:`merge_safety_fields` — union ``safety_flags``, monotonic
     ``current_risk``) and retries — so a concurrent write can never clobber a
     ``safety_flags``/``current_risk`` escalation (the clinically load-bearing race);
  2. ``SessionManager`` intentionally stays synchronous (its in-process dict is the
     per-request working copy → aliasing preserved); durability is driven from the
     already-async orchestrator, which no-ops when no store is injected;
  3. NOTE (deferred to B.3 / Stage 3): ``session_number`` is still assigned from the
     in-process ``SessionManager`` counter rather than the durable ``get_session_count``
     — two workers starting a user's first session concurrently could collide; and
     live multi-instance concurrency + real-Redis behavior are verified on staging.
Tracked under REV-16 / C.1. Unit-verified with fakeredis (``test_redis_session_repository.py``,
``test_session_durability_wiring.py``); real-Redis multi-worker verification is B.3.

Architecture Layer: Infrastructure
Principles: Repository Pattern, Dependency Inversion
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..domain.models import SessionState
from ..domain.session_manager import SessionRepository
from ..schemas import (
    HomeworkDTO,
    RiskLevel,
    SessionPhase,
    TechniqueDTO,
)

logger = structlog.get_logger(__name__)


class RedisSessionRepositorySettings(BaseSettings):
    """Session store tuning. The connection comes from the runtime's ``REDIS_URL`` client."""
    session_ttl: int = Field(default=86400)
    key_prefix: str = Field(default="solace:therapy:")
    model_config = SettingsConfigDict(
        env_prefix="THERAPY_SESSION_STORE_", env_file=".env", extra="ignore"
    )


class SessionPersistConflict(RuntimeError):
    """A CAS write-through could not converge within its retry budget (REV-16)."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        super().__init__(f"session persist did not converge (CAS contention): {session_id}")


def _session_to_dict(session: SessionState) -> dict[str, Any]:
    """Serialize a ``SessionState`` to a JSON-safe dict.

    Nested pydantic DTOs are dumped in JSON mode (UUID/datetime -> str);
    UUID/datetime/Enum scalars on the dataclass are converted explicitly so
    nothing is lost to a stringly-typed default serializer.
    """
    return {
        "session_id": str(session.session_id),
        "user_id": str(session.user_id),
        "treatment_plan_id": str(session.treatment_plan_id),
        "session_number": session.session_number,
        "current_phase": session.current_phase.value,
        "started_at": session.started_at.isoformat(),
        "mood_rating": session.mood_rating,
        "agenda_items": session.agenda_items,
        "topics_covered": session.topics_covered,
        "skills_practiced": session.skills_practiced,
        "insights_gained": session.insights_gained,
        "techniques_used": [t.model_dump(mode="json") for t in session.techniques_used],
        "homework_assigned": [h.model_dump(mode="json") for h in session.homework_assigned],
        "messages": session.messages,
        "phase_history": session.phase_history,
        "current_risk": session.current_risk.value,
        "safety_flags": session.safety_flags,
        "engagement_score": session.engagement_score,
        "session_rating": session.session_rating,
        "version": session.version,
        "updated_at": session.updated_at.isoformat(),
    }


def _dict_to_session(data: dict[str, Any]) -> SessionState:
    """Reconstruct a ``SessionState`` from its serialized form."""
    return SessionState(
        session_id=UUID(data["session_id"]),
        user_id=UUID(data["user_id"]),
        treatment_plan_id=UUID(data["treatment_plan_id"]),
        session_number=data.get("session_number", 1),
        current_phase=SessionPhase(data["current_phase"]),
        started_at=datetime.fromisoformat(data["started_at"]),
        mood_rating=data.get("mood_rating"),
        agenda_items=data.get("agenda_items", []),
        topics_covered=data.get("topics_covered", []),
        skills_practiced=data.get("skills_practiced", []),
        insights_gained=data.get("insights_gained", []),
        techniques_used=[TechniqueDTO(**t) for t in data.get("techniques_used", [])],
        homework_assigned=[HomeworkDTO(**h) for h in data.get("homework_assigned", [])],
        messages=data.get("messages", []),
        phase_history=data.get("phase_history", []),
        current_risk=RiskLevel(data["current_risk"]),
        safety_flags=data.get("safety_flags", []),
        engagement_score=data.get("engagement_score", 0.0),
        session_rating=data.get("session_rating"),
        version=data.get("version", 0),
        updated_at=(
            datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now(timezone.utc)
        ),
    )


# Ascending severity for the monotonic risk merge (NONE < LOW < ELEVATED < HIGH < CRITICAL).
_RISK_ORDER: dict[str, int] = {
    RiskLevel.NONE.value: 0,
    RiskLevel.LOW.value: 1,
    RiskLevel.ELEVATED.value: 2,
    RiskLevel.HIGH.value: 3,
    RiskLevel.CRITICAL.value: 4,
}


def max_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    """Return the higher of two risk levels — in-session escalation is monotonic,
    so a concurrent write must never silently DOWNGRADE the current risk."""
    return a if _RISK_ORDER.get(a.value, 0) >= _RISK_ORDER.get(b.value, 0) else b


def merge_safety_fields(target: SessionState, committed: SessionState) -> None:
    """Merge the clinically load-bearing fields of ``committed`` INTO ``target`` so a
    concurrent update can never LOSE a safety escalation (REV-16): ``safety_flags``
    becomes a union and ``current_risk`` is taken as the higher (never downgraded).

    Used on a CAS conflict where ``target`` carries the caller's in-flight change and
    ``committed`` is the state another worker just persisted — after the merge,
    ``target`` reflects BOTH workers' safety signals.
    """
    for flag in committed.safety_flags:
        if flag not in target.safety_flags:
            target.safety_flags.append(flag)
    target.current_risk = max_risk(target.current_risk, committed.current_risk)


class RedisSessionRepository(SessionRepository):
    """Redis-backed durable store for live therapy session state."""

    def __init__(
        self,
        settings: RedisSessionRepositorySettings | None = None,
        *,
        client: Any,
    ) -> None:
        """``client`` is the runtime's ``redis.asyncio.Redis`` (decode_responses=True).

        The store never builds or closes a connection: the runtime owns the client.
        """
        self._settings = settings or RedisSessionRepositorySettings()
        self._client: Any = client
        self._initialized = client is not None

    # -- Key helpers --

    def _session_key(self, session_id: UUID) -> str:
        return f"{self._settings.key_prefix}session:{session_id}"

    def _user_index_key(self, user_id: UUID) -> str:
        return f"{self._settings.key_prefix}user_sessions:{user_id}"

    def _user_count_key(self, user_id: UUID) -> str:
        return f"{self._settings.key_prefix}session_count:{user_id}"

    # -- SessionRepository port --

    async def save_session(self, session: SessionState) -> None:
        """Write-through persist a session (create or update)."""
        if not self._initialized:
            logger.warning("therapy_session_redis_not_initialized", op="save")
            return
        ttl = self._settings.session_ttl
        payload = json.dumps(_session_to_dict(session))
        index_key = self._user_index_key(session.user_id)
        count_key = self._user_count_key(session.user_id)

        # SADD reports 1 only when the session is newly tracked; use that to
        # increment the cumulative per-user counter exactly once per session
        # (mirrors SessionManager._user_session_counts, which never rolls back).
        added = await self._client.sadd(index_key, str(session.session_id))
        if added:
            await self._client.incr(count_key)
        await self._client.set(self._session_key(session.session_id), payload, ex=ttl)
        await self._client.expire(index_key, ttl)
        await self._client.expire(count_key, ttl)
        logger.debug("therapy_session_saved_redis", session_id=str(session.session_id))

    async def try_save_session(self, session: SessionState) -> bool:
        """Optimistic-concurrency (CAS) write-through save (REV-16).

        Persists ``session`` ONLY if the durable copy is still at the ``version``
        the caller read (checked under a Redis ``WATCH`` on the session key, so the
        check and the write are atomic). Returns ``True`` on commit (and advances
        ``session.version`` to base+1); ``False`` on a concurrent-modification
        conflict — the caller must reload the committed state, re-merge its
        load-bearing safety fields (:func:`merge_safety_fields`), and retry. A new
        session (key absent, base version 0) commits under the same WATCH and
        increments the per-user counter exactly once.
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("therapy session redis not initialized; cannot CAS-save")
        from redis.exceptions import WatchError

        key = self._session_key(session.session_id)
        index_key = self._user_index_key(session.user_id)
        count_key = self._user_count_key(session.user_id)
        ttl = self._settings.session_ttl
        base_version = session.version

        async with self._client.pipeline() as pipe:
            try:
                await pipe.watch(key)
                existing = await pipe.get(key)
                committed_version = json.loads(existing).get("version", 0) if existing else 0
                if committed_version != base_version:
                    # Someone advanced the session since we read it — conflict.
                    return False
                session.version = base_version + 1
                session.updated_at = datetime.now(timezone.utc)
                payload = json.dumps(_session_to_dict(session))
                pipe.multi()
                pipe.set(key, payload, ex=ttl)
                pipe.sadd(index_key, str(session.session_id))
                pipe.expire(index_key, ttl)
                if existing is None:
                    pipe.incr(count_key)
                    pipe.expire(count_key, ttl)
                await pipe.execute()  # WatchError if `key` changed between watch and exec
                logger.debug(
                    "therapy_session_cas_saved",
                    session_id=str(session.session_id), version=session.version,
                )
                return True
            except WatchError:
                session.version = base_version  # roll back the local bump; caller retries
                logger.info("therapy_session_cas_conflict", session_id=str(session.session_id))
                return False

    async def save_session_cas_merge(
        self, session: SessionState, max_retries: int = 5
    ) -> None:
        """Durable write-through that never loses a safety escalation (REV-16).

        Attempts a CAS save; on a concurrent-modification conflict it reloads the
        committed state, MERGES the load-bearing safety fields into ``session``
        (:func:`merge_safety_fields` — union ``safety_flags``, keep the higher
        ``current_risk``), rebases onto the committed version, and retries. Only the
        safety fields are merged; other fields remain last-writer-wins (the
        pre-existing single-object behavior). Raises :class:`SessionPersistConflict`
        if the write cannot converge within ``max_retries`` (severe contention).
        """
        for _ in range(max_retries):
            if await self.try_save_session(session):
                return
            committed = await self.get_session(session.session_id)
            if committed is not None:
                merge_safety_fields(session, committed)
                session.version = committed.version
        raise SessionPersistConflict(session.session_id)

    async def get_session(self, session_id: UUID) -> SessionState | None:
        """Read-through fetch of a session (recovers state on a fresh worker)."""
        if not self._initialized:
            return None
        data = await self._client.get(self._session_key(session_id))
        if not data:
            return None
        return _dict_to_session(json.loads(data))

    async def get_active_sessions(self, user_id: UUID | None = None) -> list[SessionState]:
        """Return stored sessions, optionally scoped to a single user."""
        if not self._initialized:
            return []
        if user_id is not None:
            session_ids = await self._client.smembers(self._user_index_key(user_id))
            keys = [self._session_key(UUID(sid)) for sid in session_ids]
        else:
            keys = []
            pattern = f"{self._settings.key_prefix}session:*"
            cursor = 0
            while True:
                cursor, batch = await self._client.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
        sessions: list[SessionState] = []
        for key in keys:
            data = await self._client.get(key)
            if data:
                sessions.append(_dict_to_session(json.loads(data)))
        return sessions

    async def delete_session(self, session_id: UUID) -> None:
        """Remove a session's durable copy (does not roll back the counter)."""
        if not self._initialized:
            return
        data = await self._client.get(self._session_key(session_id))
        await self._client.delete(self._session_key(session_id))
        if data:
            user_id = json.loads(data).get("user_id")
            if user_id:
                await self._client.srem(self._user_index_key(UUID(user_id)), str(session_id))
        logger.debug("therapy_session_deleted_redis", session_id=str(session_id))

    async def get_session_count(self, user_id: UUID) -> int:
        """Cumulative number of sessions created for a user (monotonic)."""
        if not self._initialized:
            return 0
        count = await self._client.get(self._user_count_key(user_id))
        return int(count) if count else 0

    async def delete_user_data(self, user_id: UUID) -> int:
        """GDPR right-to-erasure (REV-17): purge ALL durable state for a user.

        Reads the user's session-id index set and deletes every session key it
        references, then deletes the index set itself and the per-user session
        counter — so nothing keyed to this user survives in Redis. Returns the
        number of session records removed (bookkeeping keys are not counted).
        """
        if not self._initialized or self._client is None:
            # An erasure that silently reports 0 would claim PHI was erased when it was not.
            raise RuntimeError("therapy session redis not initialized; cannot erase user data")
        index_key = self._user_index_key(user_id)
        session_ids = await self._client.smembers(index_key)
        deleted = 0
        for sid in session_ids:
            removed = await self._client.delete(self._session_key(UUID(sid)))
            deleted += int(removed or 0)
        # Remove the index set and the cumulative counter (both keyed to the user).
        await self._client.delete(index_key)
        await self._client.delete(self._user_count_key(user_id))
        logger.info(
            "therapy_session_user_data_deleted_redis",
            user_id=str(user_id), deleted=deleted,
        )
        return deleted


def build_session_store(
    client: Any,
    settings: RedisSessionRepositorySettings | None = None,
) -> RedisSessionRepository | None:
    """Build the session store on the runtime's Redis client; `None` when there is no client."""
    if client is None:
        return None
    return RedisSessionRepository(settings, client=client)
