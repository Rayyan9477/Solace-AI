"""Unit tests for Configuration Service - API Module."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import SecretStr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services"))

from config_service.src.settings import (
    ConfigServiceSettings, ConfigurationManager, ApplicationConfig
)
from config_service.src.secrets import SecretsManager, SecretsSettings
from config_service.src.feature_flags import (
    FeatureFlagManager, FeatureFlagSettings, FeatureFlag, FlagStatus, RolloutStrategy
)
from config_service.src.api import (
    ConfigResponse, SecretResponse, FeatureFlagResponse,
    ConfigSectionResponse, HealthResponse, FlagEvaluationRequest,
    BulkEvaluationRequest, SecretRequest, FeatureFlagRequest,
)


class TestConfigResponse:
    """Tests for ConfigResponse model."""

    def test_config_response(self) -> None:
        response = ConfigResponse(
            key="test.key",
            value="test_value",
            environment="development",
            cached=True
        )
        assert response.key == "test.key"
        assert response.value == "test_value"
        assert response.environment == "development"
        assert response.cached is True

    def test_config_response_default_cached(self) -> None:
        response = ConfigResponse(key="key", value="val", environment="dev")
        assert response.cached is False

    def test_config_response_with_complex_value(self) -> None:
        response = ConfigResponse(
            key="database.config",
            value={"host": "localhost", "port": 5432},
            environment="staging"
        )
        assert response.value == {"host": "localhost", "port": 5432}


class TestConfigSectionResponse:
    """Tests for ConfigSectionResponse model."""

    def test_section_response(self) -> None:
        response = ConfigSectionResponse(
            section="database",
            data={"host": "localhost", "port": 5432},
            environment="production"
        )
        assert response.section == "database"
        assert response.data["host"] == "localhost"


class TestSecretRequest:
    """Tests for SecretRequest model."""

    def test_secret_request(self) -> None:
        request = SecretRequest(value="my-secret-value")
        assert request.value == "my-secret-value"
        assert request.metadata is None

    def test_secret_request_with_metadata(self) -> None:
        request = SecretRequest(value="secret", metadata={"env": "prod"})
        assert request.metadata == {"env": "prod"}


class TestSecretResponse:
    """Tests for SecretResponse model."""

    def test_secret_response(self) -> None:
        response = SecretResponse(
            name="api-key",
            provider="local",
            version="1",
            created_at=datetime.now(timezone.utc)
        )
        assert response.name == "api-key"
        assert response.provider == "local"

    def test_secret_response_with_updated_at(self) -> None:
        now = datetime.now(timezone.utc)
        response = SecretResponse(
            name="db-password",
            provider="vault",
            version="2",
            created_at=now,
            updated_at=now
        )
        assert response.updated_at == now


class TestFeatureFlagRequest:
    """Tests for FeatureFlagRequest model."""

    def test_minimal_request(self) -> None:
        request = FeatureFlagRequest(key="new-feature", name="New Feature")
        assert request.key == "new-feature"
        assert request.name == "New Feature"
        assert request.status == FlagStatus.DISABLED
        assert request.strategy == RolloutStrategy.NONE

    def test_full_request(self) -> None:
        request = FeatureFlagRequest(
            key="dark-mode",
            name="Dark Mode",
            description="Enable dark mode UI",
            status=FlagStatus.ENABLED,
            strategy=RolloutStrategy.PERCENTAGE,
            percentage=50,
            allowed_users=["admin-1"],
            tags=["ui", "beta"]
        )
        assert request.percentage == 50
        assert "admin-1" in request.allowed_users
        assert "ui" in request.tags


class TestFeatureFlagResponse:
    """Tests for FeatureFlagResponse model."""

    def test_flag_response(self) -> None:
        now = datetime.now(timezone.utc)
        response = FeatureFlagResponse(
            key="test-flag",
            name="Test Flag",
            description="A test flag",
            status=FlagStatus.ENABLED,
            strategy=RolloutStrategy.ALL,
            default_value=False,
            percentage=0,
            created_at=now,
            updated_at=now,
            owner="admin",
            tags=["beta"]
        )
        assert response.key == "test-flag"
        assert response.status == FlagStatus.ENABLED
        assert response.owner == "admin"


class TestFlagEvaluationRequest:
    """Tests for FlagEvaluationRequest model."""

    def test_minimal_request(self) -> None:
        request = FlagEvaluationRequest()
        assert request.user_id is None
        assert request.context == {}

    def test_with_user_and_context(self) -> None:
        request = FlagEvaluationRequest(
            user_id="user-123",
            context={"tier": "premium", "country": "US"}
        )
        assert request.user_id == "user-123"
        assert request.context["tier"] == "premium"


class TestBulkEvaluationRequest:
    """Tests for BulkEvaluationRequest model."""

    def test_bulk_request(self) -> None:
        request = BulkEvaluationRequest(keys=["flag-1", "flag-2", "flag-3"])
        assert len(request.keys) == 3
        assert request.user_id is None

    def test_bulk_request_with_context(self) -> None:
        request = BulkEvaluationRequest(
            keys=["feature-a", "feature-b"],
            user_id="user-456",
            context={"subscription": "pro"}
        )
        assert request.user_id == "user-456"


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response(self) -> None:
        response = HealthResponse(
            status="healthy",
            service="config-service",
            environment="development",
            timestamp=datetime.now(timezone.utc),
            components={
                "database": {"status": "healthy"},
                "cache": {"status": "healthy"}
            }
        )
        assert response.status == "healthy"
        assert response.service == "config-service"
        assert "database" in response.components


class TestAPIModelIntegration:
    """Integration tests for API models with core services."""

    @pytest.fixture
    def config_manager(self) -> ConfigurationManager:
        return ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))

    @pytest.fixture
    def secrets_manager(self) -> SecretsManager:
        return SecretsManager(SecretsSettings(cache_enabled=False))

    @pytest.fixture
    def flag_manager(self) -> FeatureFlagManager:
        return FeatureFlagManager(FeatureFlagSettings(evaluation_logging=False))

    @pytest.mark.asyncio
    async def test_config_response_from_manager(
        self, config_manager: ConfigurationManager
    ) -> None:
        await config_manager.load()
        value = config_manager.get("environment")
        response = ConfigResponse(
            key="environment",
            value=value,
            environment=config_manager.environment.value,
            cached=True
        )
        assert response.value == "development"

    @pytest.mark.asyncio
    async def test_secret_response_from_manager(
        self, secrets_manager: SecretsManager
    ) -> None:
        meta = await secrets_manager.set_secret("test-api-secret", SecretStr("value"))
        response = SecretResponse(
            name=meta.name,
            provider=meta.provider.value,
            version=meta.version,
            created_at=meta.created_at
        )
        assert response.name == "test-api-secret"

    @pytest.mark.asyncio
    async def test_flag_response_from_manager(
        self, flag_manager: FeatureFlagManager
    ) -> None:
        flag = FeatureFlag(
            key="api-test-flag",
            name="API Test Flag",
            status=FlagStatus.ENABLED,
            strategy=RolloutStrategy.ALL
        )
        await flag_manager.register_flag(flag)
        retrieved = await flag_manager.get_flag("api-test-flag")
        assert retrieved is not None
        response = FeatureFlagResponse(
            key=retrieved.key,
            name=retrieved.name,
            description=retrieved.description,
            status=retrieved.status,
            strategy=retrieved.strategy,
            default_value=retrieved.default_value,
            percentage=retrieved.percentage,
            created_at=retrieved.created_at,
            updated_at=retrieved.updated_at,
            owner=retrieved.owner,
            tags=retrieved.tags
        )
        assert response.status == FlagStatus.ENABLED

    @pytest.mark.asyncio
    async def test_bulk_evaluation_request_with_flags(
        self, flag_manager: FeatureFlagManager
    ) -> None:
        await flag_manager.register_flag(FeatureFlag(
            key="bulk-test-1", name="B1", status=FlagStatus.ENABLED, strategy=RolloutStrategy.ALL
        ))
        await flag_manager.register_flag(FeatureFlag(
            key="bulk-test-2", name="B2", status=FlagStatus.DISABLED
        ))
        request = BulkEvaluationRequest(keys=["bulk-test-1", "bulk-test-2"])
        results = await flag_manager.bulk_evaluate(request.keys, request.user_id, request.context)
        assert results["bulk-test-1"] is True
        assert results["bulk-test-2"] is False

    @pytest.mark.asyncio
    async def test_flag_evaluation_request_with_context(
        self, flag_manager: FeatureFlagManager
    ) -> None:
        from config_service.src.feature_flags import TargetingGroup, TargetingRule, TargetingOperator
        flag = FeatureFlag(
            key="context-flag",
            name="Context Flag",
            status=FlagStatus.CONDITIONAL,
            targeting_groups=[
                TargetingGroup(rules=[
                    TargetingRule(attribute="tier", operator=TargetingOperator.EQUALS, value="premium")
                ])
            ]
        )
        await flag_manager.register_flag(flag)
        request = FlagEvaluationRequest(user_id="user-1", context={"tier": "premium"})
        result = await flag_manager.evaluate("context-flag", request.user_id, request.context)
        assert result.enabled is True
