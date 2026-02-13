"""Unit tests for Alembic migration runner."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from solace_infrastructure.database.migrations_runner import (
    MigrationRunner,
    MigrationSettings,
    MigrationDirection,
    MigrationState,
    MigrationInfo,
    MigrationResult,
    create_migration_runner,
)


class TestMigrationDirection:
    """Tests for MigrationDirection enum."""

    def test_upgrade_value(self) -> None:
        """Test upgrade direction value."""
        assert MigrationDirection.UP.value == "upgrade"

    def test_downgrade_value(self) -> None:
        """Test downgrade direction value."""
        assert MigrationDirection.DOWN.value == "downgrade"


class TestMigrationState:
    """Tests for MigrationState enum."""

    def test_pending_state(self) -> None:
        """Test pending state value."""
        assert MigrationState.PENDING.value == "pending"

    def test_running_state(self) -> None:
        """Test running state value."""
        assert MigrationState.RUNNING.value == "running"

    def test_completed_state(self) -> None:
        """Test completed state value."""
        assert MigrationState.COMPLETED.value == "completed"

    def test_failed_state(self) -> None:
        """Test failed state value."""
        assert MigrationState.FAILED.value == "failed"

    def test_rolled_back_state(self) -> None:
        """Test rolled back state value."""
        assert MigrationState.ROLLED_BACK.value == "rolled_back"


class TestMigrationSettings:
    """Tests for MigrationSettings."""

    def test_default_database_url(self) -> None:
        """Test default database URL is set."""
        settings = MigrationSettings(database_url="postgresql://localhost/test")
        assert "postgresql" in settings.database_url

    def test_default_migrations_path(self) -> None:
        """Test default migrations path."""
        settings = MigrationSettings(database_url="postgresql://localhost/test")
        assert settings.migrations_path == "migrations"

    def test_default_auto_upgrade_disabled(self) -> None:
        """Test auto upgrade is disabled by default."""
        settings = MigrationSettings(database_url="postgresql://localhost/test")
        assert settings.auto_upgrade is False

    def test_pre_migration_check_enabled(self) -> None:
        """Test pre-migration check is enabled by default."""
        settings = MigrationSettings(database_url="postgresql://localhost/test")
        assert settings.pre_migration_check is True

    def test_post_migration_validate_enabled(self) -> None:
        """Test post-migration validation is enabled by default."""
        settings = MigrationSettings(database_url="postgresql://localhost/test")
        assert settings.post_migration_validate is True

    def test_lock_timeout_default(self) -> None:
        """Test lock timeout default value."""
        settings = MigrationSettings(database_url="postgresql://localhost/test")
        assert settings.lock_timeout_seconds == 30


class TestMigrationInfo:
    """Tests for MigrationInfo dataclass."""

    def test_migration_info_creation(self) -> None:
        """Test creating MigrationInfo."""
        info = MigrationInfo(
            revision="abc123",
            down_revision="def456",
            description="Test migration",
            created_at=datetime.now(timezone.utc),
        )
        assert info.revision == "abc123"
        assert info.down_revision == "def456"
        assert info.description == "Test migration"
        assert info.is_head is False
        assert info.is_current is False

    def test_migration_info_head_flag(self) -> None:
        """Test MigrationInfo with is_head flag."""
        info = MigrationInfo(
            revision="head123",
            down_revision=None,
            description="Head migration",
            created_at=datetime.now(timezone.utc),
            is_head=True,
        )
        assert info.is_head is True

    def test_migration_info_current_flag(self) -> None:
        """Test MigrationInfo with is_current flag."""
        info = MigrationInfo(
            revision="current123",
            down_revision="prev456",
            description="Current migration",
            created_at=datetime.now(timezone.utc),
            is_current=True,
        )
        assert info.is_current is True


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_migration_result_success(self) -> None:
        """Test successful migration result."""
        result = MigrationResult(
            state=MigrationState.COMPLETED,
            direction=MigrationDirection.UP,
            from_revision="abc123",
            to_revision="def456",
            duration_ms=100.5,
        )
        assert result.state == MigrationState.COMPLETED
        assert result.direction == MigrationDirection.UP
        assert result.from_revision == "abc123"
        assert result.to_revision == "def456"
        assert result.duration_ms == 100.5
        assert result.error is None

    def test_migration_result_failure(self) -> None:
        """Test failed migration result."""
        result = MigrationResult(
            state=MigrationState.FAILED,
            direction=MigrationDirection.UP,
            from_revision="abc123",
            to_revision=None,
            error="Connection failed",
        )
        assert result.state == MigrationState.FAILED
        assert result.to_revision is None
        assert result.error == "Connection failed"

    def test_migration_result_default_timestamp(self) -> None:
        """Test default executed_at timestamp is set."""
        result = MigrationResult(
            state=MigrationState.COMPLETED,
            direction=MigrationDirection.UP,
            from_revision=None,
            to_revision="abc123",
        )
        assert result.executed_at is not None
        assert isinstance(result.executed_at, datetime)


class TestMigrationRunner:
    """Tests for MigrationRunner class."""

    def test_runner_initialization(self) -> None:
        """Test MigrationRunner can be initialized."""
        runner = MigrationRunner(settings=MigrationSettings(database_url="postgresql://localhost/test"))
        assert runner is not None

    def test_runner_with_settings(self) -> None:
        """Test MigrationRunner with custom settings."""
        settings = MigrationSettings(database_url="postgresql://localhost/test", auto_upgrade=True)
        runner = MigrationRunner(settings=settings)
        assert runner._settings.auto_upgrade is True

    def test_register_hook_valid_event(self) -> None:
        """Test registering a valid hook."""
        runner = MigrationRunner(settings=MigrationSettings(database_url="postgresql://localhost/test"))

        async def test_hook() -> None:
            pass

        runner.register_hook("pre_upgrade", test_hook)
        assert test_hook in runner._hooks["pre_upgrade"]

    def test_register_hook_invalid_event(self) -> None:
        """Test registering an invalid hook raises error."""
        runner = MigrationRunner(settings=MigrationSettings(database_url="postgresql://localhost/test"))

        async def test_hook() -> None:
            pass

        with pytest.raises(ValueError, match="Invalid hook event"):
            runner.register_hook("invalid_event", test_hook)

    def test_hooks_initialized_empty(self) -> None:
        """Test hooks are initialized as empty lists."""
        runner = MigrationRunner(settings=MigrationSettings(database_url="postgresql://localhost/test"))
        assert "pre_upgrade" in runner._hooks
        assert "post_upgrade" in runner._hooks
        assert "pre_downgrade" in runner._hooks
        assert "post_downgrade" in runner._hooks
        assert len(runner._hooks["pre_upgrade"]) == 0


class TestCreateMigrationRunner:
    """Tests for create_migration_runner factory function."""

    @pytest.mark.asyncio
    async def test_create_migration_runner_returns_runner(self) -> None:
        """Test factory function returns MigrationRunner."""
        with patch.object(MigrationRunner, "initialize", new_callable=AsyncMock):
            runner = await create_migration_runner(MigrationSettings(database_url="postgresql://localhost/test"))
            assert isinstance(runner, MigrationRunner)

    @pytest.mark.asyncio
    async def test_create_migration_runner_with_settings(self) -> None:
        """Test factory function accepts settings."""
        settings = MigrationSettings(database_url="postgresql://localhost/test", auto_upgrade=True)
        with patch.object(MigrationRunner, "initialize", new_callable=AsyncMock):
            runner = await create_migration_runner(settings)
            assert runner._settings.auto_upgrade is True
