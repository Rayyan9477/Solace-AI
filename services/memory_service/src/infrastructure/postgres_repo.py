"""
Solace-AI Memory Service - PostgreSQL Repository.
Async repository for structured memory storage using SQLAlchemy and asyncpg.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID, uuid4

import structlog
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    and_,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

logger = structlog.get_logger(__name__)

metadata = MetaData()

memory_records = Table(
    "memory_records", metadata,
    Column("id", PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
    Column("user_id", PG_UUID(as_uuid=True), nullable=False, index=True),
    Column("session_id", PG_UUID(as_uuid=True), nullable=True, index=True),
    Column("tier", String(50), nullable=False, index=True),
    Column("content", Text, nullable=False),
    Column("content_type", String(50), default="message"),
    Column("retention_category", String(50), default="medium_term"),
    Column("importance_score", Numeric(5, 4), default=Decimal("0.5")),
    Column("emotional_valence", Numeric(5, 4), nullable=True),
    Column("retention_strength", Numeric(5, 4), default=Decimal("1.0")),
    Column("metadata", JSON, default=dict),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(UTC)),
    Column("accessed_at", DateTime(timezone=True), default=lambda: datetime.now(UTC)),
    Column("is_archived", Boolean, default=False),
)

session_summaries = Table(
    "session_summaries", metadata,
    Column("id", PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
    Column("session_id", PG_UUID(as_uuid=True), nullable=False, unique=True),
    Column("user_id", PG_UUID(as_uuid=True), nullable=False, index=True),
    Column("session_number", Numeric, default=1),
    Column("summary_text", Text, nullable=False),
    Column("key_topics", JSON, default=list),
    Column("emotional_arc", JSON, default=list),
    Column("techniques_used", JSON, default=list),
    Column("key_insights", JSON, default=list),
    Column("message_count", Numeric, default=0),
    Column("duration_minutes", Numeric, default=0),
    Column("session_date", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(UTC)),
    Column("retention_strength", Numeric(5, 4), default=Decimal("1.0")),
)

user_facts = Table(
    "user_facts", metadata,
    Column("id", PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
    Column("user_id", PG_UUID(as_uuid=True), nullable=False, index=True),
    Column("category", String(50), nullable=False, index=True),
    Column("content", Text, nullable=False),
    Column("confidence", Numeric(5, 4), default=Decimal("0.7")),
    Column("importance", Numeric(5, 4), default=Decimal("0.5")),
    Column("status", String(50), default="active"),
    Column("source_session_id", PG_UUID(as_uuid=True), nullable=True),
    Column("version", Numeric, default=1),
    Column("supersedes", PG_UUID(as_uuid=True), nullable=True),
    Column("verified_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(UTC)),
    Column("updated_at", DateTime(timezone=True), default=lambda: datetime.now(UTC)),
    Column("related_entities", JSON, default=list),
    Column("metadata", JSON, default=dict),
)

therapeutic_events = Table(
    "therapeutic_events", metadata,
    Column("id", PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
    Column("user_id", PG_UUID(as_uuid=True), nullable=False, index=True),
    Column("session_id", PG_UUID(as_uuid=True), nullable=True, index=True),
    Column("event_type", String(50), nullable=False, index=True),
    Column("severity", String(50), default="medium"),
    Column("title", String(255), nullable=False),
    Column("description", Text, default=""),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("ingested_at", DateTime(timezone=True), default=lambda: datetime.now(UTC)),
    Column("valid_from", DateTime(timezone=True), nullable=True),
    Column("valid_to", DateTime(timezone=True), nullable=True),
    Column("related_events", JSON, default=list),
    Column("payload", JSON, default=dict),
    Column("retention_strength", Numeric(5, 4), default=Decimal("1.0")),
)


class PostgresSettings(BaseSettings):
    """PostgreSQL connection configuration."""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="solace_memory", description="Database name")
    user: str = Field(default="solace", description="Database user")
    password: SecretStr = Field(default="", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=5, description="Max pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout seconds")
    echo_sql: bool = Field(default=False, description="Echo SQL statements")
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    @property
    def connection_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.database}"


class MemoryRecordProtocol(Protocol):
    """Protocol for memory record data."""
    record_id: UUID
    user_id: UUID
    session_id: UUID | None
    tier: str
    content: str
    content_type: str
    retention_category: str
    importance_score: Decimal
    metadata: dict[str, Any]
    created_at: datetime


class PostgresRepository:
    """Async PostgreSQL repository for memory storage."""

    def __init__(self, settings: PostgresSettings | None = None) -> None:
        self._settings = settings or PostgresSettings()
        engine_kwargs: dict[str, Any] = {
            "echo": self._settings.echo_sql,
        }
        if self._settings.pool_size == 0:
            engine_kwargs["poolclass"] = NullPool
        else:
            engine_kwargs["pool_size"] = self._settings.pool_size
            engine_kwargs["max_overflow"] = self._settings.max_overflow
            engine_kwargs["pool_timeout"] = self._settings.pool_timeout
        self._engine = create_async_engine(self._settings.connection_url, **engine_kwargs)
        self._session_factory = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
        self._background_tasks: set[asyncio.Task] = set()
        self._stats = {"inserts": 0, "updates": 0, "deletes": 0, "queries": 0}

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Provide a transactional session scope."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def initialize(self) -> None:
        """Verify database connectivity. Table creation is handled by Alembic migrations."""
        async with self._engine.begin() as conn:
            # Table creation is now managed by Alembic migrations in the centralized ORM.
            # Previously: await conn.run_sync(metadata.create_all)
            await conn.execute(select(func.now()))
        logger.info("postgres_initialized", database=self._settings.database)

    async def close(self) -> None:
        """Close database connections."""
        await self._engine.dispose()
        logger.info("postgres_closed")

    async def store_memory_record(self, record: MemoryRecordProtocol) -> UUID:
        """Store a memory record."""
        start = time.perf_counter()
        self._stats["inserts"] += 1
        async with self.session() as session:
            stmt = insert(memory_records).values(
                id=record.record_id, user_id=record.user_id,
                session_id=record.session_id, tier=record.tier,
                content=record.content, content_type=record.content_type,
                retention_category=record.retention_category,
                importance_score=record.importance_score,
                metadata=record.metadata, created_at=record.created_at,
            )
            await session.execute(stmt)
        logger.debug("memory_record_stored", record_id=str(record.record_id),
                     time_ms=int((time.perf_counter() - start) * 1000))
        return record.record_id

    async def get_memory_record(self, record_id: UUID) -> dict[str, Any] | None:
        """Get a memory record by ID."""
        self._stats["queries"] += 1
        async with self.session() as session:
            stmt = select(memory_records).where(memory_records.c.id == record_id)
            result = await session.execute(stmt)
            row = result.fetchone()
            if row:
                # Tracked background task for access time update
                task = asyncio.create_task(self._update_access_time(record_id))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                return dict(row._mapping)
        return None

    async def _update_access_time(self, record_id: UUID) -> None:
        """Update accessed_at timestamp in a separate transaction."""
        try:
            async with self.session() as session:
                await session.execute(
                    update(memory_records).where(memory_records.c.id == record_id)
                    .values(accessed_at=datetime.now(UTC))
                )
        except Exception:
            logger.debug("access_time_update_failed", record_id=str(record_id))

    async def get_user_records(self, user_id: UUID, tier: str | None = None,
                                limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Get memory records for a user with optional tier filter."""
        self._stats["queries"] += 1
        async with self.session() as session:
            stmt = select(memory_records).where(
                and_(memory_records.c.user_id == user_id, memory_records.c.is_archived == False)
            )
            if tier:
                stmt = stmt.where(memory_records.c.tier == tier)
            stmt = stmt.order_by(memory_records.c.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(stmt)
            return [dict(row._mapping) for row in result.fetchall()]

    async def get_session_records(self, session_id: UUID, limit: int = 500) -> list[dict[str, Any]]:
        """Get all records for a session."""
        self._stats["queries"] += 1
        async with self.session() as session:
            stmt = select(memory_records).where(memory_records.c.session_id == session_id) \
                .order_by(memory_records.c.created_at.asc()).limit(limit)
            result = await session.execute(stmt)
            return [dict(row._mapping) for row in result.fetchall()]

    async def update_retention_strength(self, record_id: UUID, new_strength: Decimal) -> bool:
        """Update retention strength for decay model."""
        self._stats["updates"] += 1
        async with self.session() as session:
            stmt = update(memory_records).where(memory_records.c.id == record_id) \
                .values(retention_strength=new_strength)
            result = await session.execute(stmt)
            return result.rowcount > 0

    async def archive_record(self, record_id: UUID) -> bool:
        """Archive a memory record."""
        self._stats["updates"] += 1
        async with self.session() as session:
            stmt = update(memory_records).where(memory_records.c.id == record_id) \
                .values(is_archived=True)
            result = await session.execute(stmt)
            return result.rowcount > 0

    async def delete_record(self, record_id: UUID) -> bool:
        """Permanently delete a memory record."""
        self._stats["deletes"] += 1
        async with self.session() as session:
            stmt = delete(memory_records).where(memory_records.c.id == record_id)
            result = await session.execute(stmt)
            return result.rowcount > 0

    async def store_session_summary(self, summary_data: dict[str, Any]) -> UUID:
        """Store a session summary."""
        self._stats["inserts"] += 1
        summary_id = summary_data.get("summary_id", uuid4())
        async with self.session() as session:
            stmt = insert(session_summaries).values(id=summary_id, **{
                k: v for k, v in summary_data.items() if k != "summary_id"
            })
            await session.execute(stmt)
        logger.debug("session_summary_stored", summary_id=str(summary_id))
        return summary_id

    async def get_user_summaries(self, user_id: UUID, limit: int = 10) -> list[dict[str, Any]]:
        """Get session summaries for a user."""
        self._stats["queries"] += 1
        async with self.session() as session:
            stmt = select(session_summaries).where(session_summaries.c.user_id == user_id) \
                .order_by(session_summaries.c.session_date.desc()).limit(limit)
            result = await session.execute(stmt)
            return [dict(row._mapping) for row in result.fetchall()]

    async def store_user_fact(self, fact_data: dict[str, Any]) -> UUID:
        """Store a user fact."""
        self._stats["inserts"] += 1
        fact_id = fact_data.get("fact_id", uuid4())
        async with self.session() as session:
            stmt = insert(user_facts).values(id=fact_id, **{
                k: v for k, v in fact_data.items() if k != "fact_id"
            })
            await session.execute(stmt)
        logger.debug("user_fact_stored", fact_id=str(fact_id))
        return fact_id

    async def get_user_facts(self, user_id: UUID, category: str | None = None,
                              active_only: bool = True) -> list[dict[str, Any]]:
        """Get facts for a user."""
        self._stats["queries"] += 1
        async with self.session() as session:
            stmt = select(user_facts).where(user_facts.c.user_id == user_id)
            if category:
                stmt = stmt.where(user_facts.c.category == category)
            if active_only:
                stmt = stmt.where(user_facts.c.status == "active")
            stmt = stmt.order_by(user_facts.c.importance.desc())
            result = await session.execute(stmt)
            return [dict(row._mapping) for row in result.fetchall()]

    async def store_therapeutic_event(self, event_data: dict[str, Any]) -> UUID:
        """Store a therapeutic event."""
        self._stats["inserts"] += 1
        event_id = event_data.get("event_id", uuid4())
        async with self.session() as session:
            stmt = insert(therapeutic_events).values(id=event_id, **{
                k: v for k, v in event_data.items() if k != "event_id"
            })
            await session.execute(stmt)
        logger.debug("therapeutic_event_stored", event_id=str(event_id))
        return event_id

    async def get_user_events(self, user_id: UUID, event_type: str | None = None,
                               start_date: datetime | None = None,
                               end_date: datetime | None = None,
                               limit: int = 50) -> list[dict[str, Any]]:
        """Get therapeutic events for a user."""
        self._stats["queries"] += 1
        async with self.session() as session:
            stmt = select(therapeutic_events).where(therapeutic_events.c.user_id == user_id)
            if event_type:
                stmt = stmt.where(therapeutic_events.c.event_type == event_type)
            if start_date:
                stmt = stmt.where(therapeutic_events.c.occurred_at >= start_date)
            if end_date:
                stmt = stmt.where(therapeutic_events.c.occurred_at <= end_date)
            stmt = stmt.order_by(therapeutic_events.c.occurred_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return [dict(row._mapping) for row in result.fetchall()]

    async def get_crisis_events(self, user_id: UUID) -> list[dict[str, Any]]:
        """Get all crisis events for a user (never decay)."""
        return await self.get_user_events(user_id, event_type="crisis", limit=100)

    async def apply_batch_decay(self, user_id: UUID, decay_factor: Decimal,
                                  exclude_permanent: bool = True) -> int:
        """Apply exponential decay to all user records in batch.

        Uses Ebbinghaus formula: retention = retention_strength * exp(-rate * hours_since_last_access)
        where rate is the decay_factor per hour.
        """
        self._stats["updates"] += 1
        async with self.session() as session:
            conditions = [
                memory_records.c.user_id == user_id,
                memory_records.c.is_archived == False,
            ]
            if exclude_permanent:
                conditions.append(memory_records.c.retention_category != "permanent")
            # Exponential decay based on hours since last access/update
            hours_elapsed = func.extract(
                "epoch",
                func.now() - func.coalesce(memory_records.c.accessed_at, memory_records.c.created_at),
            ) / 3600.0
            stmt = update(memory_records).where(and_(*conditions)).values(
                retention_strength=func.greatest(
                    Decimal("0.0"),
                    memory_records.c.retention_strength * func.exp(-decay_factor * hours_elapsed),
                )
            )
            result = await session.execute(stmt)
            return result.rowcount

    async def get_max_session_number(self, user_id: UUID) -> int | None:
        """Get the maximum session number for a user (for session count recovery)."""
        self._stats["queries"] += 1
        async with self.session() as session:
            stmt = select(func.max(session_summaries.c.session_number)).where(
                session_summaries.c.user_id == user_id
            )
            result = await session.execute(stmt)
            row = result.scalar()
            return int(row) if row is not None else None

    async def delete_user_data(self, user_id: UUID) -> tuple[int, int, int, int]:
        """Delete all data for a user (GDPR compliance)."""
        async with self.session() as session:
            r1 = await session.execute(delete(memory_records).where(memory_records.c.user_id == user_id))
            r2 = await session.execute(delete(session_summaries).where(session_summaries.c.user_id == user_id))
            r3 = await session.execute(delete(user_facts).where(user_facts.c.user_id == user_id))
            r4 = await session.execute(delete(therapeutic_events).where(therapeutic_events.c.user_id == user_id))
        logger.info("user_data_deleted", user_id=str(user_id))
        return r1.rowcount, r2.rowcount, r3.rowcount, r4.rowcount

    def get_statistics(self) -> dict[str, Any]:
        """Get repository statistics."""
        return {**self._stats, "database": self._settings.database, "host": self._settings.host}
