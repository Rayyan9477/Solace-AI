"""Solace-AI Alembic Migration Runner - Database schema management.

Provides enterprise-grade database migration capabilities:
- Programmatic Alembic execution
- Migration history tracking
- Rollback support with safety checks
- Pre/post migration hooks
- Health validation after migrations
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Awaitable

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
import structlog

from solace_common.exceptions import DatabaseError, ConfigurationError

logger = structlog.get_logger(__name__)


class MigrationDirection(str, Enum):
    """Direction of migration execution."""
    UP = "upgrade"
    DOWN = "downgrade"


class MigrationState(str, Enum):
    """Current state of migration process."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationSettings(BaseSettings):
    """Migration configuration from environment."""

    database_url: str = Field(
        description="Database URL for migrations (required via MIGRATION_DATABASE_URL env var)"
    )
    migrations_path: str = Field(default="migrations")
    alembic_ini_path: str = Field(default="alembic.ini")
    script_location: str = Field(default="migrations/versions")
    auto_upgrade: bool = Field(default=False)
    pre_migration_check: bool = Field(default=True)
    post_migration_validate: bool = Field(default=True)
    lock_timeout_seconds: int = Field(default=30, ge=1, le=300)

    model_config = SettingsConfigDict(
        env_prefix="MIGRATION_",
        env_file=".env",
        extra="ignore",
    )


@dataclass
class MigrationInfo:
    """Information about a single migration revision."""

    revision: str
    down_revision: str | None
    description: str
    created_at: datetime
    is_head: bool = False
    is_current: bool = False


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    state: MigrationState
    direction: MigrationDirection
    from_revision: str | None
    to_revision: str | None
    executed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_ms: float = 0.0
    error: str | None = None
    revisions_applied: list[str] = field(default_factory=list)


class MigrationRunner:
    """Orchestrates database migrations with safety checks and logging."""

    def __init__(
        self,
        settings: MigrationSettings | None = None,
        engine: AsyncEngine | None = None,
    ) -> None:
        self._settings = settings or MigrationSettings()
        self._engine = engine
        self._alembic_config: Config | None = None
        self._hooks: dict[str, list[Callable[[], Awaitable[None]]]] = {
            "pre_upgrade": [],
            "post_upgrade": [],
            "pre_downgrade": [],
            "post_downgrade": [],
        }

    async def initialize(self) -> None:
        """Initialize migration runner with database connection."""
        if self._engine is None:
            self._engine = create_async_engine(
                self._settings.database_url,
                pool_pre_ping=True,
                echo=False,
            )
        self._alembic_config = self._create_alembic_config()
        logger.info("migration_runner_initialized")

    def _create_alembic_config(self) -> Config:
        """Create Alembic configuration programmatically."""
        config = Config()
        config.set_main_option("script_location", self._settings.script_location)
        config.set_main_option("sqlalchemy.url", self._settings.database_url)
        return config

    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("migration_runner_closed")

    @asynccontextmanager
    async def _acquire_lock(self) -> AsyncIterator[bool]:
        """Acquire advisory lock to prevent concurrent migrations."""
        lock_id = 123456789
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(f"SELECT pg_try_advisory_lock({lock_id})")
            )
            acquired = result.scalar()
            if not acquired:
                raise DatabaseError(
                    "Could not acquire migration lock - another migration may be running",
                    operation="acquire_lock",
                )
            try:
                yield True
            finally:
                await conn.execute(
                    text(f"SELECT pg_advisory_unlock({lock_id})")
                )

    async def get_current_revision(self) -> str | None:
        """Get the current database revision."""
        async with self._engine.begin() as conn:
            def get_revision(sync_conn: Any) -> str | None:
                context = MigrationContext.configure(sync_conn)
                return context.get_current_revision()
            return await conn.run_sync(get_revision)

    async def get_head_revision(self) -> str | None:
        """Get the latest available migration revision."""
        if not self._alembic_config:
            raise ConfigurationError("Migration runner not initialized")
        script = ScriptDirectory.from_config(self._alembic_config)
        return script.get_current_head()

    async def get_pending_migrations(self) -> list[MigrationInfo]:
        """Get list of migrations pending application."""
        current = await self.get_current_revision()
        if not self._alembic_config:
            raise ConfigurationError("Migration runner not initialized")
        script = ScriptDirectory.from_config(self._alembic_config)
        pending: list[MigrationInfo] = []
        for rev in script.walk_revisions():
            if current and rev.revision == current:
                break
            pending.append(
                MigrationInfo(
                    revision=rev.revision,
                    down_revision=rev.down_revision,
                    description=rev.doc or "",
                    created_at=datetime.now(timezone.utc),
                    is_head=rev.revision == script.get_current_head(),
                )
            )
        return list(reversed(pending))

    async def get_migration_history(self) -> list[MigrationInfo]:
        """Get list of applied migrations."""
        if not self._alembic_config:
            raise ConfigurationError("Migration runner not initialized")
        current = await self.get_current_revision()
        script = ScriptDirectory.from_config(self._alembic_config)
        history: list[MigrationInfo] = []
        for rev in script.walk_revisions():
            info = MigrationInfo(
                revision=rev.revision,
                down_revision=rev.down_revision,
                description=rev.doc or "",
                created_at=datetime.now(timezone.utc),
                is_head=rev.revision == script.get_current_head(),
                is_current=rev.revision == current,
            )
            history.append(info)
            if current and rev.revision == current:
                break
        return history

    def register_hook(
        self,
        event: str,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """Register a pre/post migration hook."""
        if event not in self._hooks:
            raise ValueError(f"Invalid hook event: {event}")
        self._hooks[event].append(callback)

    async def _run_hooks(self, event: str) -> None:
        """Execute registered hooks for an event."""
        for hook in self._hooks.get(event, []):
            await hook()

    async def upgrade(
        self,
        target: str = "head",
        dry_run: bool = False,
    ) -> MigrationResult:
        """Upgrade database to target revision."""
        start_time = asyncio.get_running_loop().time()
        from_rev = await self.get_current_revision()

        try:
            async with self._acquire_lock():
                if self._settings.pre_migration_check:
                    await self._pre_migration_check()
                await self._run_hooks("pre_upgrade")
                if not dry_run:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        command.upgrade,
                        self._alembic_config,
                        target,
                    )
                await self._run_hooks("post_upgrade")
                if self._settings.post_migration_validate:
                    await self._post_migration_validate()
                to_rev = await self.get_current_revision()
                duration = (asyncio.get_running_loop().time() - start_time) * 1000
                result = MigrationResult(
                    state=MigrationState.COMPLETED,
                    direction=MigrationDirection.UP,
                    from_revision=from_rev,
                    to_revision=to_rev,
                    duration_ms=duration,
                )
                logger.info(
                    "migration_upgrade_completed",
                    from_rev=from_rev,
                    to_rev=to_rev,
                    duration_ms=duration,
                )
                return result
        except Exception as e:
            logger.error("migration_upgrade_failed", error=str(e))
            return MigrationResult(
                state=MigrationState.FAILED,
                direction=MigrationDirection.UP,
                from_revision=from_rev,
                to_revision=None,
                error=str(e),
            )

    async def downgrade(
        self,
        target: str = "-1",
        dry_run: bool = False,
    ) -> MigrationResult:
        """Downgrade database to target revision."""
        start_time = asyncio.get_running_loop().time()
        from_rev = await self.get_current_revision()

        try:
            async with self._acquire_lock():
                await self._run_hooks("pre_downgrade")
                if not dry_run:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        command.downgrade,
                        self._alembic_config,
                        target,
                    )
                await self._run_hooks("post_downgrade")
                to_rev = await self.get_current_revision()
                duration = (asyncio.get_running_loop().time() - start_time) * 1000
                result = MigrationResult(
                    state=MigrationState.COMPLETED,
                    direction=MigrationDirection.DOWN,
                    from_revision=from_rev,
                    to_revision=to_rev,
                    duration_ms=duration,
                )
                logger.info(
                    "migration_downgrade_completed",
                    from_rev=from_rev,
                    to_rev=to_rev,
                )
                return result
        except Exception as e:
            logger.error("migration_downgrade_failed", error=str(e))
            return MigrationResult(
                state=MigrationState.FAILED,
                direction=MigrationDirection.DOWN,
                from_revision=from_rev,
                to_revision=None,
                error=str(e),
            )

    async def _pre_migration_check(self) -> None:
        """Validate database connectivity before migration."""
        async with self._engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            if result.scalar() != 1:
                raise DatabaseError(
                    "Pre-migration check failed: database not responsive",
                    operation="pre_check",
                )

    async def _post_migration_validate(self) -> None:
        """Validate database state after migration."""
        async with self._engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            if result.scalar() != 1:
                raise DatabaseError(
                    "Post-migration validation failed",
                    operation="post_validate",
                )

    async def check_health(self) -> dict[str, Any]:
        """Check migration system health."""
        try:
            current = await self.get_current_revision()
            head = await self.get_head_revision()
            pending = await self.get_pending_migrations()
            return {
                "status": "healthy",
                "current_revision": current,
                "head_revision": head,
                "pending_count": len(pending),
                "is_up_to_date": current == head,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


async def create_migration_runner(
    settings: MigrationSettings | None = None,
) -> MigrationRunner:
    """Factory function to create and initialize a migration runner."""
    runner = MigrationRunner(settings)
    await runner.initialize()
    return runner
