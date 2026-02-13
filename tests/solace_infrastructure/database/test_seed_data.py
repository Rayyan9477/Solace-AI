"""Unit tests for seed data loader."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from solace_infrastructure.database.seed_data import (
    SeedDataLoader,
    SeedSettings,
    SeedResult,
    SeedBatch,
    SeedCategory,
    Environment,
    BaseSeedProvider,
    SystemConfigSeedProvider,
    ClinicalReferenceSeedProvider,
    TherapyTechniqueSeedProvider,
    SafetyResourceSeedProvider,
    create_seed_loader,
)


class TestEnvironment:
    """Tests for Environment enum."""

    def test_development_value(self) -> None:
        """Test development environment value."""
        assert Environment.DEVELOPMENT.value == "development"

    def test_staging_value(self) -> None:
        """Test staging environment value."""
        assert Environment.STAGING.value == "staging"

    def test_production_value(self) -> None:
        """Test production environment value."""
        assert Environment.PRODUCTION.value == "production"

    def test_test_value(self) -> None:
        """Test test environment value."""
        assert Environment.TEST.value == "test"


class TestSeedCategory:
    """Tests for SeedCategory enum."""

    def test_system_category(self) -> None:
        """Test system category value."""
        assert SeedCategory.SYSTEM.value == "system"

    def test_clinical_category(self) -> None:
        """Test clinical category value."""
        assert SeedCategory.CLINICAL.value == "clinical"

    def test_reference_category(self) -> None:
        """Test reference category value."""
        assert SeedCategory.REFERENCE.value == "reference"

    def test_test_category(self) -> None:
        """Test test category value."""
        assert SeedCategory.TEST.value == "test"


class TestSeedSettings:
    """Tests for SeedSettings."""

    def test_default_database_url(self) -> None:
        """Test default database URL is set."""
        settings = SeedSettings(database_url="postgresql://localhost/test")
        assert "postgresql" in settings.database_url

    def test_default_environment(self) -> None:
        """Test default environment is development."""
        settings = SeedSettings(database_url="postgresql://localhost/test")
        assert settings.environment == Environment.DEVELOPMENT

    def test_force_reseed_disabled(self) -> None:
        """Test force reseed is disabled by default."""
        settings = SeedSettings(database_url="postgresql://localhost/test")
        assert settings.force_reseed is False

    def test_validate_after_seed_enabled(self) -> None:
        """Test validation is enabled by default."""
        settings = SeedSettings(database_url="postgresql://localhost/test")
        assert settings.validate_after_seed is True

    def test_default_batch_size(self) -> None:
        """Test default batch size."""
        settings = SeedSettings(database_url="postgresql://localhost/test")
        assert settings.batch_size == 100


class TestSeedResult:
    """Tests for SeedResult dataclass."""

    def test_successful_result(self) -> None:
        """Test successful seed result."""
        result = SeedResult(
            category=SeedCategory.SYSTEM,
            table_name="test_table",
            records_created=10,
            records_skipped=5,
            duration_ms=100.5,
            success=True,
        )
        assert result.category == SeedCategory.SYSTEM
        assert result.table_name == "test_table"
        assert result.records_created == 10
        assert result.records_skipped == 5
        assert result.success is True
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed seed result."""
        result = SeedResult(
            category=SeedCategory.CLINICAL,
            table_name="clinical_data",
            records_created=0,
            records_skipped=0,
            duration_ms=50.0,
            success=False,
            error="Connection error",
        )
        assert result.success is False
        assert result.error == "Connection error"


class TestSeedBatch:
    """Tests for SeedBatch dataclass."""

    def test_seed_batch_creation(self) -> None:
        """Test creating a seed batch."""
        batch = SeedBatch(
            table_name="test_table",
            category=SeedCategory.SYSTEM,
            data=[{"key": "value"}],
            unique_keys=["key"],
        )
        assert batch.table_name == "test_table"
        assert batch.category == SeedCategory.SYSTEM
        assert len(batch.data) == 1
        assert batch.unique_keys == ["key"]

    def test_seed_batch_default_values(self) -> None:
        """Test seed batch default values."""
        batch = SeedBatch(
            table_name="test",
            category=SeedCategory.TEST,
            data=[],
        )
        assert batch.unique_keys == []
        assert batch.dependencies == []


class TestSystemConfigSeedProvider:
    """Tests for SystemConfigSeedProvider."""

    def test_category_is_system(self) -> None:
        """Test category is SYSTEM."""
        provider = SystemConfigSeedProvider()
        assert provider.category == SeedCategory.SYSTEM

    def test_table_name(self) -> None:
        """Test table name is correct."""
        provider = SystemConfigSeedProvider()
        assert provider.table_name == "system_configurations"

    def test_unique_keys(self) -> None:
        """Test unique keys include key column."""
        provider = SystemConfigSeedProvider()
        assert "key" in provider.unique_keys

    def test_get_data_returns_list(self) -> None:
        """Test get_data returns a list."""
        provider = SystemConfigSeedProvider()
        data = provider.get_data(Environment.DEVELOPMENT)
        assert isinstance(data, list)

    def test_get_data_includes_required_fields(self) -> None:
        """Test get_data includes required fields."""
        provider = SystemConfigSeedProvider()
        data = provider.get_data(Environment.DEVELOPMENT)
        assert len(data) > 0
        for record in data:
            assert "id" in record
            assert "key" in record
            assert "value" in record

    def test_development_includes_debug_config(self) -> None:
        """Test development includes debug configuration."""
        provider = SystemConfigSeedProvider()
        data = provider.get_data(Environment.DEVELOPMENT)
        keys = [d["key"] for d in data]
        assert "debug.verbose_logging" in keys


class TestClinicalReferenceSeedProvider:
    """Tests for ClinicalReferenceSeedProvider."""

    def test_category_is_clinical(self) -> None:
        """Test category is CLINICAL."""
        provider = ClinicalReferenceSeedProvider()
        assert provider.category == SeedCategory.CLINICAL

    def test_table_name(self) -> None:
        """Test table name is correct."""
        provider = ClinicalReferenceSeedProvider()
        assert provider.table_name == "clinical_references"

    def test_get_data_includes_dsm_codes(self) -> None:
        """Test get_data includes DSM-5 codes."""
        provider = ClinicalReferenceSeedProvider()
        data = provider.get_data(Environment.PRODUCTION)
        codes = [d["code"] for d in data]
        assert "F32.0" in codes
        assert "F41.1" in codes


class TestTherapyTechniqueSeedProvider:
    """Tests for TherapyTechniqueSeedProvider."""

    def test_category_is_clinical(self) -> None:
        """Test category is CLINICAL."""
        provider = TherapyTechniqueSeedProvider()
        assert provider.category == SeedCategory.CLINICAL

    def test_table_name(self) -> None:
        """Test table name is correct."""
        provider = TherapyTechniqueSeedProvider()
        assert provider.table_name == "therapy_techniques"

    def test_get_data_includes_techniques(self) -> None:
        """Test get_data includes therapy techniques."""
        provider = TherapyTechniqueSeedProvider()
        data = provider.get_data(Environment.PRODUCTION)
        modalities = [d["modality"] for d in data]
        assert "CBT" in modalities
        assert "DBT" in modalities
        assert "ACT" in modalities


class TestSafetyResourceSeedProvider:
    """Tests for SafetyResourceSeedProvider."""

    def test_category_is_system(self) -> None:
        """Test category is SYSTEM."""
        provider = SafetyResourceSeedProvider()
        assert provider.category == SeedCategory.SYSTEM

    def test_table_name(self) -> None:
        """Test table name is correct."""
        provider = SafetyResourceSeedProvider()
        assert provider.table_name == "safety_resources"

    def test_get_data_includes_988_lifeline(self) -> None:
        """Test get_data includes 988 lifeline."""
        provider = SafetyResourceSeedProvider()
        data = provider.get_data(Environment.PRODUCTION)
        codes = [d["resource_code"] for d in data]
        assert "988_LIFELINE" in codes


class TestSeedDataLoader:
    """Tests for SeedDataLoader class."""

    def test_loader_initialization(self) -> None:
        """Test SeedDataLoader can be initialized."""
        loader = SeedDataLoader(settings=SeedSettings(database_url="postgresql://localhost/test"))
        assert loader is not None

    def test_loader_with_settings(self) -> None:
        """Test SeedDataLoader with custom settings."""
        settings = SeedSettings(database_url="postgresql://localhost/test", environment=Environment.STAGING)
        loader = SeedDataLoader(settings=settings)
        assert loader._settings.environment == Environment.STAGING

    def test_register_provider(self) -> None:
        """Test registering a custom provider."""
        loader = SeedDataLoader(settings=SeedSettings(database_url="postgresql://localhost/test"))
        provider = SystemConfigSeedProvider()
        initial_count = len(loader._providers)
        loader.register_provider(provider)
        assert len(loader._providers) == initial_count + 1


class TestCreateSeedLoader:
    """Tests for create_seed_loader factory function."""

    @pytest.mark.asyncio
    async def test_create_seed_loader_returns_loader(self) -> None:
        """Test factory function returns SeedDataLoader."""
        with patch.object(SeedDataLoader, "initialize", new_callable=AsyncMock):
            loader = await create_seed_loader(SeedSettings(database_url="postgresql://localhost/test"))
            assert isinstance(loader, SeedDataLoader)

    @pytest.mark.asyncio
    async def test_create_seed_loader_with_settings(self) -> None:
        """Test factory function accepts settings."""
        settings = SeedSettings(database_url="postgresql://localhost/test", environment=Environment.TEST)
        with patch.object(SeedDataLoader, "initialize", new_callable=AsyncMock):
            loader = await create_seed_loader(settings)
            assert loader._settings.environment == Environment.TEST
