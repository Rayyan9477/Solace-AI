"""Unit tests for Configuration Service - Secrets Module."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from pydantic import SecretStr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services"))

from config_service.src.secrets import (
    SecretProvider,
    SecretMetadata,
    CachedSecret,
    SecretsSettings,
    SecretAccessEvent,
    LocalSecretsProvider,
    SecretsManager,
    create_secrets_manager,
)


class TestSecretProvider:
    """Tests for SecretProvider enum."""

    def test_provider_values(self) -> None:
        assert SecretProvider.VAULT.value == "vault"
        assert SecretProvider.AWS_SECRETS_MANAGER.value == "aws_secrets_manager"
        assert SecretProvider.LOCAL.value == "local"

    def test_provider_from_string(self) -> None:
        provider = SecretProvider("local")
        assert provider == SecretProvider.LOCAL


class TestSecretMetadata:
    """Tests for SecretMetadata model."""

    def test_default_metadata(self) -> None:
        meta = SecretMetadata(name="test-secret", provider=SecretProvider.LOCAL)
        assert meta.name == "test-secret"
        assert meta.provider == SecretProvider.LOCAL
        assert meta.version == "1"
        assert meta.rotation_enabled is False
        assert meta.rotation_days is None
        assert isinstance(meta.created_at, datetime)

    def test_metadata_with_rotation(self) -> None:
        meta = SecretMetadata(
            name="rotating-secret",
            provider=SecretProvider.VAULT,
            rotation_enabled=True,
            rotation_days=30,
        )
        assert meta.rotation_enabled is True
        assert meta.rotation_days == 30

    def test_metadata_with_tags(self) -> None:
        meta = SecretMetadata(
            name="tagged-secret",
            provider=SecretProvider.AWS_SECRETS_MANAGER,
            tags={"env": "production", "team": "platform"},
        )
        assert meta.tags == {"env": "production", "team": "platform"}


class TestCachedSecret:
    """Tests for CachedSecret model."""

    def test_cached_secret_not_expired(self) -> None:
        cached = CachedSecret(
            value=SecretStr("secret-value"),
            metadata=SecretMetadata(name="test", provider=SecretProvider.LOCAL),
            ttl_seconds=3600,
        )
        assert cached.is_expired is False

    def test_cached_secret_expired(self) -> None:
        cached = CachedSecret(
            value=SecretStr("secret-value"),
            metadata=SecretMetadata(name="test", provider=SecretProvider.LOCAL),
            cached_at=datetime.now(timezone.utc) - timedelta(hours=2),
            ttl_seconds=60,
        )
        assert cached.is_expired is True


class TestSecretsSettings:
    """Tests for SecretsSettings model."""

    def test_default_settings(self) -> None:
        settings = SecretsSettings()
        assert settings.default_provider == SecretProvider.LOCAL
        assert settings.cache_enabled is True
        assert settings.cache_ttl_seconds == 300
        assert settings.vault_url == "http://localhost:8200"
        assert settings.vault_mount_path == "secret"
        assert settings.aws_region == "us-east-1"
        assert settings.audit_enabled is True

    def test_settings_with_vault_token(self) -> None:
        settings = SecretsSettings(vault_token=SecretStr("hvs.token123"))
        assert settings.vault_token is not None
        assert settings.vault_token.get_secret_value() == "hvs.token123"


class TestSecretAccessEvent:
    """Tests for SecretAccessEvent model."""

    def test_access_event_success(self) -> None:
        event = SecretAccessEvent(
            secret_name="api-key",
            operation="get",
            provider=SecretProvider.LOCAL,
            success=True,
            user_id="user-123",
        )
        assert event.secret_name == "api-key"
        assert event.operation == "get"
        assert event.success is True
        assert event.user_id == "user-123"
        assert isinstance(event.timestamp, datetime)

    def test_access_event_failure(self) -> None:
        event = SecretAccessEvent(
            secret_name="missing-secret",
            operation="get",
            provider=SecretProvider.VAULT,
            success=False,
            error_message="Secret not found",
        )
        assert event.success is False
        assert event.error_message == "Secret not found"


class TestLocalSecretsProvider:
    """Tests for LocalSecretsProvider."""

    @pytest.fixture
    def provider(self) -> LocalSecretsProvider:
        return LocalSecretsProvider(SecretsSettings())

    @pytest.mark.asyncio
    async def test_set_and_get_secret(self, provider: LocalSecretsProvider) -> None:
        await provider.set_secret("test-key", SecretStr("test-value"))
        value, meta = await provider.get_secret("test-key")
        assert value.get_secret_value() == "test-value"
        assert meta.name == "test-key"
        assert meta.provider == SecretProvider.LOCAL

    @pytest.mark.asyncio
    async def test_get_nonexistent_secret(self, provider: LocalSecretsProvider) -> None:
        with pytest.raises(KeyError):
            await provider.get_secret("nonexistent")

    @pytest.mark.asyncio
    async def test_delete_secret(self, provider: LocalSecretsProvider) -> None:
        await provider.set_secret("to-delete", SecretStr("value"))
        deleted = await provider.delete_secret("to-delete")
        assert deleted is True
        with pytest.raises(KeyError):
            await provider.get_secret("to-delete")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_secret(self, provider: LocalSecretsProvider) -> None:
        deleted = await provider.delete_secret("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_secrets(self, provider: LocalSecretsProvider) -> None:
        await provider.set_secret("secret-1", SecretStr("value1"))
        await provider.set_secret("secret-2", SecretStr("value2"))
        secrets = await provider.list_secrets()
        assert len(secrets) == 2
        names = [s.name for s in secrets]
        assert "secret-1" in names
        assert "secret-2" in names

    @pytest.mark.asyncio
    async def test_list_secrets_with_prefix(self, provider: LocalSecretsProvider) -> None:
        await provider.set_secret("app/db-password", SecretStr("value1"))
        await provider.set_secret("app/api-key", SecretStr("value2"))
        await provider.set_secret("other/key", SecretStr("value3"))
        secrets = await provider.list_secrets(prefix="app/")
        assert len(secrets) == 2

    @pytest.mark.asyncio
    async def test_check_health(self, provider: LocalSecretsProvider) -> None:
        health = await provider.check_health()
        assert health["status"] == "healthy"
        assert health["provider"] == "local"


class TestSecretsManager:
    """Tests for SecretsManager."""

    @pytest.fixture
    def manager(self) -> SecretsManager:
        return SecretsManager(SecretsSettings(cache_enabled=True, cache_ttl_seconds=60))

    @pytest.mark.asyncio
    async def test_set_and_get_secret(self, manager: SecretsManager) -> None:
        await manager.set_secret("test-secret", SecretStr("secret-value"))
        value = await manager.get_secret("test-secret")
        assert value.get_secret_value() == "secret-value"

    @pytest.mark.asyncio
    async def test_get_secret_with_caching(self, manager: SecretsManager) -> None:
        await manager.set_secret("cached-secret", SecretStr("value"))
        value1 = await manager.get_secret("cached-secret")
        value2 = await manager.get_secret("cached-secret")
        assert value1.get_secret_value() == value2.get_secret_value()
        assert len(manager._cache) > 0

    @pytest.mark.asyncio
    async def test_get_secret_bypass_cache(self, manager: SecretsManager) -> None:
        await manager.set_secret("bypass-secret", SecretStr("value"))
        await manager.get_secret("bypass-secret")
        assert len(manager._cache) > 0
        await manager.get_secret("bypass-secret", bypass_cache=True)

    @pytest.mark.asyncio
    async def test_delete_secret(self, manager: SecretsManager) -> None:
        await manager.set_secret("to-delete", SecretStr("value"))
        deleted = await manager.delete_secret("to-delete")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_list_secrets(self, manager: SecretsManager) -> None:
        await manager.set_secret("list-1", SecretStr("v1"))
        await manager.set_secret("list-2", SecretStr("v2"))
        secrets = await manager.list_secrets()
        assert len(secrets) >= 2

    @pytest.mark.asyncio
    async def test_check_health(self, manager: SecretsManager) -> None:
        health = await manager.check_health()
        # Status can be healthy or degraded depending on provider availability
        assert health["status"] in ["healthy", "degraded"]
        assert "providers" in health
        # Local provider should always be healthy
        assert "local" in health["providers"]
        assert health["providers"]["local"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_audit_logging(self, manager: SecretsManager) -> None:
        await manager.set_secret("audited-secret", SecretStr("value"))
        await manager.get_secret("audited-secret")
        assert len(manager._audit_buffer) > 0

    @pytest.mark.asyncio
    async def test_invalid_provider(self, manager: SecretsManager) -> None:
        with pytest.raises(ValueError, match="Provider not configured"):
            await manager.get_secret("test", provider=SecretProvider.AZURE_KEY_VAULT)


class TestCreateSecretsManager:
    """Tests for create_secrets_manager factory function."""

    def test_create_secrets_manager_default(self) -> None:
        manager = create_secrets_manager()
        assert isinstance(manager, SecretsManager)

    def test_create_secrets_manager_with_settings(self) -> None:
        settings = SecretsSettings(cache_ttl_seconds=120)
        manager = create_secrets_manager(settings)
        assert manager._settings.cache_ttl_seconds == 120
