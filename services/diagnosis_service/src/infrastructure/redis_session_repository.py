"""Solace-AI Diagnosis Service - Redis live-session store (REV-16).

The diagnosis ``DiagnosisService`` held live session state (phase, extracted
symptoms, differential, reasoning history, safety_flags) in an in-memory
``_active_sessions`` dict that was lost on restart and never shared across
workers — only COMPLETED records were persisted (at ``end_session``). This module
adds a durable, concurrency-safe Redis store for the LIVE session, mirroring the
therapy service's REV-16 store:

  * whole-object JSON serialization of the diagnosis ``SessionState`` (pydantic
    DTOs dumped in JSON mode), keyed by session id with a TTL;
  * optimistic concurrency: a ``version`` field + :meth:`try_save_session` (version
    check under Redis ``WATCH`` -> ``MULTI/EXEC``);
  * :meth:`save_session_cas_merge` — on a conflict, reload the committed state,
    UNION the ``safety_flags`` (:func:`merge_safety_flags`) and retry, so a
    concurrent write can never clobber a safety escalation.

Wired into ``DiagnosisService`` via ``_persist_session`` / ``_load_session``.
Live multi-instance / real-Redis verification is a Stage-3 (B.3) task.
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
from ..schemas import DiagnosisPhase, DifferentialDTO, ReasoningStepResultDTO, SymptomDTO

logger = structlog.get_logger(__name__)


class DiagnosisSessionRedisSettings(BaseSettings):
    """Store settings for the diagnosis live-session Redis store.

    The connection comes from the runtime (``REDIS_URL``); only the TTL and key prefix live here.
    """

    session_ttl: int = Field(default=86400)  # 24h
    key_prefix: str = Field(default="solace:diagnosis:")
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_SESSION_STORE_", extra="ignore")


def _session_to_dict(session: SessionState) -> dict[str, Any]:
    """Serialize a diagnosis ``SessionState`` to a JSON-safe dict."""
    return {
        "session_id": str(session.session_id),
        "user_id": str(session.user_id),
        "session_number": session.session_number,
        "phase": session.phase.value,
        "symptoms": [s.model_dump(mode="json") for s in session.symptoms],
        "differential": session.differential.model_dump(mode="json") if session.differential else None,
        "messages": session.messages,
        "started_at": session.started_at.isoformat(),
        "reasoning_history": [r.model_dump(mode="json") for r in session.reasoning_history],
        "safety_flags": session.safety_flags,
        "version": session.version,
        "updated_at": session.updated_at.isoformat(),
    }


def _dict_to_session(data: dict[str, Any]) -> SessionState:
    """Reconstruct a diagnosis ``SessionState`` from its serialized form."""
    return SessionState(
        session_id=UUID(data["session_id"]),
        user_id=UUID(data["user_id"]),
        session_number=data.get("session_number", 1),
        phase=DiagnosisPhase(data["phase"]),
        symptoms=[SymptomDTO.model_validate(s) for s in data.get("symptoms", [])],
        differential=(
            DifferentialDTO.model_validate(data["differential"])
            if data.get("differential")
            else None
        ),
        messages=data.get("messages", []),
        started_at=datetime.fromisoformat(data["started_at"]),
        reasoning_history=[
            ReasoningStepResultDTO.model_validate(r) for r in data.get("reasoning_history", [])
        ],
        safety_flags=data.get("safety_flags", []),
        version=data.get("version", 0),
        updated_at=(
            datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now(timezone.utc)
        ),
    )


def merge_safety_flags(target: SessionState, committed: SessionState) -> None:
    """Union the ``safety_flags`` of ``committed`` INTO ``target`` so a concurrent
    update can never LOSE a safety signal (REV-16). Diagnosis carries no risk-level
    scalar, so only the flag set is merged."""
    for flag in committed.safety_flags:
        if flag not in target.safety_flags:
            target.safety_flags.append(flag)


class SessionPersistConflict(RuntimeError):
    """A CAS write-through could not converge within its retry budget (REV-16)."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        super().__init__(f"session persist did not converge (CAS contention): {session_id}")


class DiagnosisSessionStore:
    """Redis-backed durable store for live diagnosis session state (REV-16)."""

    def __init__(
        self,
        settings: DiagnosisSessionRedisSettings | None = None,
        *,
        client: Any,
    ) -> None:
        """``client`` is the runtime's Redis client; the store never opens or closes a connection."""
        self._settings = settings or DiagnosisSessionRedisSettings()
        self._client: Any = client
        self._initialized = client is not None

    def _session_key(self, session_id: UUID) -> str:
        return f"{self._settings.key_prefix}session:{session_id}"

    def _user_index_key(self, user_id: UUID) -> str:
        return f"{self._settings.key_prefix}user_sessions:{user_id}"

    def _user_count_key(self, user_id: UUID) -> str:
        return f"{self._settings.key_prefix}session_count:{user_id}"

    async def try_save_session(self, session: SessionState) -> bool:
        """Optimistic-concurrency (CAS) write-through save. Persists ONLY if the
        durable copy is still at the ``version`` the caller read (checked under a
        Redis WATCH). Returns True on commit (advancing ``session.version``); False
        on a concurrent-modification conflict."""
        if not self._initialized or self._client is None:
            raise RuntimeError("diagnosis session redis not initialized; cannot CAS-save")
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
                await pipe.execute()
                logger.debug("diagnosis_session_cas_saved", session_id=str(session.session_id), version=session.version)
                return True
            except WatchError:
                session.version = base_version
                logger.info("diagnosis_session_cas_conflict", session_id=str(session.session_id))
                return False

    async def save_session_cas_merge(self, session: SessionState, max_retries: int = 5) -> None:
        """Durable write-through that never loses a safety flag (REV-16). On a CAS
        conflict, reload committed state, UNION the ``safety_flags`` into ``session``,
        rebase onto the committed version, and retry. Raises
        :class:`SessionPersistConflict` if it cannot converge."""
        for _ in range(max_retries):
            if await self.try_save_session(session):
                return
            committed = await self.get_session(session.session_id)
            if committed is not None:
                merge_safety_flags(session, committed)
                session.version = committed.version
        raise SessionPersistConflict(session.session_id)

    async def get_session(self, session_id: UUID) -> SessionState | None:
        """Read-through fetch of a live session (recovers state on a fresh worker)."""
        if not self._initialized or self._client is None:
            return None
        data = await self._client.get(self._session_key(session_id))
        if not data:
            return None
        return _dict_to_session(json.loads(data))

    async def delete_session(self, session_id: UUID) -> None:
        """Evict a session (and its user-index entry) from the store."""
        if not self._initialized or self._client is None:
            return
        existing = await self._client.get(self._session_key(session_id))
        await self._client.delete(self._session_key(session_id))
        if existing:
            try:
                user_id = json.loads(existing).get("user_id")
                if user_id:
                    await self._client.srem(self._user_index_key(UUID(user_id)), str(session_id))
            except (ValueError, TypeError):
                pass

    async def get_session_count(self, user_id: UUID) -> int:
        """Cumulative number of sessions created for a user (monotonic)."""
        if not self._initialized or self._client is None:
            return 0
        count = await self._client.get(self._user_count_key(user_id))
        return int(count) if count else 0

    async def delete_user_data(self, user_id: UUID) -> int:
        """GDPR right-to-erasure (REV-17): purge ALL of a user's live-session PHI.

        The live-session blobs hold special-category PHI (symptoms, differential,
        safety_flags), so a GDPR erasure MUST remove them. Reads the user's session
        index, deletes every referenced session blob, then deletes the index set and
        the per-user session counter. Fail-loud: an operational Redis error RAISES
        (never a silent 0 that would let the orchestrator attest a complete erasure
        while PHI remains). Accepts the ``str`` the orchestrator passes.
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("diagnosis session redis not initialized; cannot erase")
        uid = user_id if isinstance(user_id, UUID) else UUID(str(user_id))
        index_key = self._user_index_key(uid)
        session_ids = await self._client.smembers(index_key)
        deleted = 0
        for sid in session_ids:
            removed = await self._client.delete(self._session_key(UUID(sid)))
            deleted += int(removed or 0)
        # Remove the index set and the cumulative counter (both keyed to the user).
        await self._client.delete(index_key)
        await self._client.delete(self._user_count_key(uid))
        logger.info("diagnosis_session_user_data_deleted", user_id=str(uid), deleted=deleted)
        return deleted


def build_session_store(
    client: Any,
    settings: DiagnosisSessionRedisSettings | None = None,
) -> DiagnosisSessionStore | None:
    """Build the session store on the runtime's Redis client; `None` when there is no client."""
    if client is None:
        return None
    return DiagnosisSessionStore(settings, client=client)
