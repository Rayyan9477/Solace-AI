"""Unit tests for Kafka Schema Registry Management module."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from solace_infrastructure.kafka.schemas import (
    SchemaType,
    CompatibilityLevel,
    SchemaFormat,
    SchemaVersion,
    SchemaMetadata,
    CompatibilityResult,
    SchemaRegistrySettings,
    SchemaDefinition,
    JsonSchemaValidator,
    AvroSchemaValidator,
    SchemaCache,
    SchemaRegistryAdapter,
    SchemaManager,
    create_schema_manager,
)


class TestSchemaEnums:
    """Tests for schema-related enums."""

    def test_schema_type_values(self) -> None:
        assert SchemaType.AVRO.value == "AVRO"
        assert SchemaType.JSON.value == "JSON"
        assert SchemaType.PROTOBUF.value == "PROTOBUF"

    def test_compatibility_level_values(self) -> None:
        assert CompatibilityLevel.BACKWARD.value == "BACKWARD"
        assert CompatibilityLevel.FORWARD.value == "FORWARD"
        assert CompatibilityLevel.FULL.value == "FULL"
        assert CompatibilityLevel.NONE.value == "NONE"


class TestSchemaRegistrySettings:
    """Tests for SchemaRegistrySettings."""

    def test_default_settings(self) -> None:
        settings = SchemaRegistrySettings()
        assert settings.url == "http://localhost:8081"
        assert settings.timeout_seconds == 30
        assert settings.cache_capacity == 1000

    def test_custom_settings(self) -> None:
        settings = SchemaRegistrySettings(
            url="http://registry:8081",
            username="admin",
            password="secret",
            default_compatibility=CompatibilityLevel.FULL,
        )
        assert settings.url == "http://registry:8081"
        assert settings.username == "admin"
        assert settings.default_compatibility == CompatibilityLevel.FULL


class TestSchemaDefinition:
    """Tests for SchemaDefinition model."""

    def test_json_schema_definition(self) -> None:
        schema_str = json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        definition = SchemaDefinition(
            subject="test-value",
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        assert definition.subject == "test-value"
        assert definition.schema_type == SchemaType.JSON

    def test_fingerprint_generation(self) -> None:
        schema_str = json.dumps({"type": "object"})
        definition = SchemaDefinition(subject="test", schema_str=schema_str)
        assert len(definition.fingerprint) == 16

    def test_fingerprint_consistency(self) -> None:
        schema_str = json.dumps({"type": "object", "properties": {"a": {"type": "string"}}})
        def1 = SchemaDefinition(subject="test", schema_str=schema_str)
        def2 = SchemaDefinition(subject="test", schema_str=schema_str)
        assert def1.fingerprint == def2.fingerprint

    def test_to_registry_format(self) -> None:
        schema_str = json.dumps({"type": "object"})
        definition = SchemaDefinition(
            subject="test",
            schema_str=schema_str,
            schema_type=SchemaType.JSON,
        )
        registry_format = definition.to_registry_format()
        assert registry_format["schema"] == schema_str
        assert registry_format["schemaType"] == "JSON"


class TestJsonSchemaValidator:
    """Tests for JsonSchemaValidator."""

    @pytest.fixture
    def validator(self) -> JsonSchemaValidator:
        return JsonSchemaValidator()

    def test_validate_valid_schema(self, validator: JsonSchemaValidator) -> None:
        schema_str = json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        valid, error = validator.validate(schema_str)
        assert valid is True
        assert error is None

    def test_validate_invalid_json(self, validator: JsonSchemaValidator) -> None:
        valid, error = validator.validate("{invalid json}")
        assert valid is False
        assert "Invalid JSON" in error

    def test_validate_missing_type(self, validator: JsonSchemaValidator) -> None:
        schema_str = json.dumps({"description": "no type"})
        valid, error = validator.validate(schema_str)
        assert valid is False
        assert "type" in error.lower()

    def test_validate_non_object_schema(self, validator: JsonSchemaValidator) -> None:
        schema_str = json.dumps("just a string")
        valid, error = validator.validate(schema_str)
        assert valid is False
        assert "object" in error.lower()

    def test_check_compatibility_backward(self, validator: JsonSchemaValidator) -> None:
        old_schema = json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        new_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })
        result = validator.check_compatibility(new_schema, old_schema)
        assert result.is_compatible is True
        assert any("Added property" in msg for msg in result.messages)

    def test_check_compatibility_breaking_removal(self, validator: JsonSchemaValidator) -> None:
        old_schema = json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
        })
        new_schema = json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        result = validator.check_compatibility(new_schema, old_schema)
        assert result.is_compatible is False
        assert any("Removed" in change for change in result.breaking_changes)

    def test_check_compatibility_new_required(self, validator: JsonSchemaValidator) -> None:
        old_schema = json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        new_schema = json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["email"],
        })
        result = validator.check_compatibility(new_schema, old_schema)
        assert result.is_compatible is False
        assert any("required" in change.lower() for change in result.breaking_changes)


class TestAvroSchemaValidator:
    """Tests for AvroSchemaValidator."""

    @pytest.fixture
    def validator(self) -> AvroSchemaValidator:
        return AvroSchemaValidator()

    def test_validate_valid_record(self, validator: AvroSchemaValidator) -> None:
        schema_str = json.dumps({
            "type": "record",
            "name": "User",
            "fields": [{"name": "name", "type": "string"}],
        })
        valid, error = validator.validate(schema_str)
        assert valid is True
        assert error is None

    def test_validate_missing_type(self, validator: AvroSchemaValidator) -> None:
        schema_str = json.dumps({"name": "User"})
        valid, error = validator.validate(schema_str)
        assert valid is False
        assert "type" in error.lower()

    def test_validate_record_missing_name(self, validator: AvroSchemaValidator) -> None:
        schema_str = json.dumps({
            "type": "record",
            "fields": [{"name": "id", "type": "int"}],
        })
        valid, error = validator.validate(schema_str)
        assert valid is False
        assert "name" in error.lower()

    def test_validate_record_missing_fields(self, validator: AvroSchemaValidator) -> None:
        schema_str = json.dumps({"type": "record", "name": "User"})
        valid, error = validator.validate(schema_str)
        assert valid is False
        assert "fields" in error.lower()

    def test_check_compatibility_add_optional_field(self, validator: AvroSchemaValidator) -> None:
        old_schema = json.dumps({
            "type": "record", "name": "User",
            "fields": [{"name": "name", "type": "string"}],
        })
        new_schema = json.dumps({
            "type": "record", "name": "User",
            "fields": [
                {"name": "name", "type": "string"},
                {"name": "age", "type": "int", "default": 0},
            ],
        })
        result = validator.check_compatibility(new_schema, old_schema)
        assert result.is_compatible is True

    def test_check_compatibility_remove_field_no_default(self, validator: AvroSchemaValidator) -> None:
        old_schema = json.dumps({
            "type": "record", "name": "User",
            "fields": [
                {"name": "name", "type": "string"},
                {"name": "email", "type": "string"},
            ],
        })
        new_schema = json.dumps({
            "type": "record", "name": "User",
            "fields": [{"name": "name", "type": "string"}],
        })
        result = validator.check_compatibility(new_schema, old_schema)
        assert result.is_compatible is False


class TestSchemaCache:
    """Tests for SchemaCache."""

    def test_put_and_get(self) -> None:
        cache = SchemaCache[str](capacity=10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing(self) -> None:
        cache = SchemaCache[str](capacity=10)
        assert cache.get("nonexistent") is None

    def test_eviction(self) -> None:
        cache = SchemaCache[str](capacity=2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        # One key should be evicted
        present = sum(1 for k in ["key1", "key2", "key3"] if cache.get(k))
        assert present == 2

    def test_invalidate(self) -> None:
        cache = SchemaCache[str](capacity=10)
        cache.put("key1", "value1")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self) -> None:
        cache = SchemaCache[str](capacity=10)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestSchemaRegistryAdapter:
    """Tests for SchemaRegistryAdapter."""

    @pytest.fixture
    def adapter(self) -> SchemaRegistryAdapter:
        settings = SchemaRegistrySettings()
        return SchemaRegistryAdapter(settings)

    @pytest.fixture
    def mock_adapter(self) -> SchemaRegistryAdapter:
        """Create adapter in mock mode (no HTTP session)."""
        settings = SchemaRegistrySettings()
        adapter = SchemaRegistryAdapter(settings)
        adapter._session = None  # Force mock mode
        return adapter

    @pytest.mark.asyncio
    async def test_connect_close(self, adapter: SchemaRegistryAdapter) -> None:
        await adapter.connect()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_register_schema_mock(self, mock_adapter: SchemaRegistryAdapter) -> None:
        # Using mock_adapter which has _session=None to avoid HTTP calls
        definition = SchemaDefinition(
            subject="test-value",
            schema_str=json.dumps({"type": "object"}),
        )
        schema_id = await mock_adapter.register_schema(definition)
        assert isinstance(schema_id, int)

    @pytest.mark.asyncio
    async def test_list_subjects_mock(self, mock_adapter: SchemaRegistryAdapter) -> None:
        # Using mock_adapter for mock mode behavior
        subjects = await mock_adapter.list_subjects()
        assert isinstance(subjects, list)

    @pytest.mark.asyncio
    async def test_check_compatibility_mock(self, mock_adapter: SchemaRegistryAdapter) -> None:
        # Using mock_adapter - returns True in mock mode
        is_compatible = await mock_adapter.check_compatibility(
            "test-value",
            json.dumps({"type": "object"}),
        )
        assert is_compatible is True


class TestSchemaManager:
    """Tests for SchemaManager."""

    @pytest.fixture
    def manager(self) -> SchemaManager:
        return SchemaManager()

    @pytest.fixture
    def mock_manager(self) -> SchemaManager:
        """Create manager with adapter in mock mode."""
        manager = SchemaManager()
        manager._adapter._session = None  # Force mock mode
        return manager

    @pytest.mark.asyncio
    async def test_connect_close(self, manager: SchemaManager) -> None:
        await manager.connect()
        await manager.close()

    @pytest.mark.asyncio
    async def test_register_valid_schema(self, mock_manager: SchemaManager) -> None:
        # Using mock_manager to avoid HTTP calls
        definition = SchemaDefinition(
            subject="test-value",
            schema_str=json.dumps({
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }),
            schema_type=SchemaType.JSON,
        )
        schema_id = await mock_manager.register(definition)
        assert isinstance(schema_id, int)

    @pytest.mark.asyncio
    async def test_register_invalid_schema(self, manager: SchemaManager) -> None:
        await manager.connect()
        definition = SchemaDefinition(
            subject="test-value",
            schema_str="not valid json",
            schema_type=SchemaType.JSON,
        )
        with pytest.raises(ValueError, match="Invalid schema"):
            await manager.register(definition)
        await manager.close()

    @pytest.mark.asyncio
    async def test_validate_evolution(self, manager: SchemaManager) -> None:
        await manager.connect()
        new_schema = json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        result = await manager.validate_evolution("new-subject", new_schema)
        # No existing schema, should be compatible
        assert result.is_compatible is True
        await manager.close()

    @pytest.mark.asyncio
    async def test_list_subjects(self, manager: SchemaManager) -> None:
        await manager.connect()
        subjects = await manager.list_subjects()
        assert isinstance(subjects, list)
        await manager.close()

    def test_get_validator(self, manager: SchemaManager) -> None:
        json_validator = manager.get_validator(SchemaType.JSON)
        assert isinstance(json_validator, JsonSchemaValidator)
        avro_validator = manager.get_validator(SchemaType.AVRO)
        assert isinstance(avro_validator, AvroSchemaValidator)


class TestSchemaVersion:
    """Tests for SchemaVersion dataclass."""

    def test_schema_version_creation(self) -> None:
        version = SchemaVersion(
            schema_id=1,
            version=1,
            schema_str='{"type": "object"}',
            schema_type=SchemaType.JSON,
            fingerprint="abc123",
        )
        assert version.schema_id == 1
        assert version.version == 1
        assert version.schema_type == SchemaType.JSON


class TestCompatibilityResult:
    """Tests for CompatibilityResult dataclass."""

    def test_compatible_result(self) -> None:
        result = CompatibilityResult(
            is_compatible=True,
            messages=["Added optional field"],
        )
        assert result.is_compatible is True
        assert len(result.messages) == 1
        assert len(result.breaking_changes) == 0

    def test_incompatible_result(self) -> None:
        result = CompatibilityResult(
            is_compatible=False,
            breaking_changes=["Removed required field"],
        )
        assert result.is_compatible is False
        assert len(result.breaking_changes) == 1


class TestFactoryFunction:
    """Tests for factory functions."""

    @pytest.mark.asyncio
    async def test_create_schema_manager(self) -> None:
        manager = await create_schema_manager()
        assert isinstance(manager, SchemaManager)
        await manager.close()
