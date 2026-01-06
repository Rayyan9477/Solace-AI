"""Unit tests for Weaviate schema manager."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from solace_infrastructure.database.weaviate_schema import (
    WeaviateSchemaManager,
    WeaviateSchemaSettings,
    SolaceCollections,
    CollectionDefinition,
    CollectionType,
    SchemaVersion,
    CURRENT_SCHEMA_VERSION,
    setup_weaviate_schema,
)
from solace_infrastructure.weaviate import VectorDistanceMetric, PropertyDataType


class TestCollectionType:
    """Tests for CollectionType enum."""

    def test_conversation_type(self) -> None:
        """Test conversation collection type."""
        assert CollectionType.CONVERSATION.value == "conversation"

    def test_session_type(self) -> None:
        """Test session collection type."""
        assert CollectionType.SESSION.value == "session"

    def test_therapeutic_type(self) -> None:
        """Test therapeutic collection type."""
        assert CollectionType.THERAPEUTIC.value == "therapeutic"

    def test_safety_type(self) -> None:
        """Test safety collection type."""
        assert CollectionType.SAFETY.value == "safety"

    def test_user_fact_type(self) -> None:
        """Test user fact collection type."""
        assert CollectionType.USER_FACT.value == "user_fact"


class TestSchemaVersion:
    """Tests for SchemaVersion enum."""

    def test_v1_0(self) -> None:
        """Test v1.0 schema version."""
        assert SchemaVersion.V1_0.value == "1.0"

    def test_v2_0(self) -> None:
        """Test v2.0 schema version."""
        assert SchemaVersion.V2_0.value == "2.0"

    def test_current_schema_version(self) -> None:
        """Test current schema version is set."""
        assert CURRENT_SCHEMA_VERSION == SchemaVersion.V2_0


class TestWeaviateSchemaSettings:
    """Tests for WeaviateSchemaSettings."""

    def test_default_replication_factor(self) -> None:
        """Test default replication factor."""
        settings = WeaviateSchemaSettings()
        assert settings.replication_factor == 1

    def test_multi_tenancy_enabled_default(self) -> None:
        """Test multi-tenancy is enabled by default."""
        settings = WeaviateSchemaSettings()
        assert settings.multi_tenancy_enabled is True

    def test_auto_create_collections_default(self) -> None:
        """Test auto create collections is enabled by default."""
        settings = WeaviateSchemaSettings()
        assert settings.auto_create_collections is True

    def test_delete_existing_disabled_default(self) -> None:
        """Test delete existing is disabled by default."""
        settings = WeaviateSchemaSettings()
        assert settings.delete_existing_on_init is False

    def test_default_embedding_dimensions(self) -> None:
        """Test default embedding dimensions."""
        settings = WeaviateSchemaSettings()
        assert settings.embedding_dimensions == 1536


class TestCollectionDefinition:
    """Tests for CollectionDefinition dataclass."""

    def test_collection_definition_creation(self) -> None:
        """Test creating a collection definition."""
        definition = CollectionDefinition(
            name="TestCollection",
            collection_type=CollectionType.CONVERSATION,
            description="Test collection",
            properties=[],
        )
        assert definition.name == "TestCollection"
        assert definition.collection_type == CollectionType.CONVERSATION
        assert definition.description == "Test collection"

    def test_collection_definition_defaults(self) -> None:
        """Test collection definition defaults."""
        definition = CollectionDefinition(
            name="Test",
            collection_type=CollectionType.SESSION,
            description="Test",
            properties=[],
        )
        assert definition.distance_metric == VectorDistanceMetric.COSINE
        assert definition.multi_tenancy is True
        assert definition.replication_factor == 1

    def test_to_config_method(self) -> None:
        """Test to_config method returns CollectionConfig."""
        definition = CollectionDefinition(
            name="TestConfig",
            collection_type=CollectionType.USER_FACT,
            description="Test config",
            properties=[],
        )
        config = definition.to_config()
        assert config.name == "TestConfig"
        assert config.description == "Test config"


class TestSolaceCollections:
    """Tests for SolaceCollections factory."""

    def test_conversation_memory_collection(self) -> None:
        """Test conversation memory collection definition."""
        collection = SolaceCollections.conversation_memory()
        assert collection.name == "ConversationMemory"
        assert collection.collection_type == CollectionType.CONVERSATION
        assert len(collection.properties) > 0

    def test_conversation_memory_has_content_property(self) -> None:
        """Test conversation memory has content property."""
        collection = SolaceCollections.conversation_memory()
        prop_names = [p.name for p in collection.properties]
        assert "content" in prop_names
        assert "user_id" in prop_names
        assert "session_id" in prop_names

    def test_session_summary_collection(self) -> None:
        """Test session summary collection definition."""
        collection = SolaceCollections.session_summary()
        assert collection.name == "SessionSummary"
        assert collection.collection_type == CollectionType.SESSION

    def test_session_summary_has_summary_property(self) -> None:
        """Test session summary has summary property."""
        collection = SolaceCollections.session_summary()
        prop_names = [p.name for p in collection.properties]
        assert "summary" in prop_names
        assert "key_topics" in prop_names

    def test_therapeutic_insight_collection(self) -> None:
        """Test therapeutic insight collection definition."""
        collection = SolaceCollections.therapeutic_insight()
        assert collection.name == "TherapeuticInsight"
        assert collection.collection_type == CollectionType.THERAPEUTIC

    def test_therapeutic_insight_has_insight_property(self) -> None:
        """Test therapeutic insight has insight property."""
        collection = SolaceCollections.therapeutic_insight()
        prop_names = [p.name for p in collection.properties]
        assert "insight" in prop_names
        assert "insight_type" in prop_names

    def test_user_fact_collection(self) -> None:
        """Test user fact collection definition."""
        collection = SolaceCollections.user_fact()
        assert collection.name == "UserFact"
        assert collection.collection_type == CollectionType.USER_FACT

    def test_user_fact_has_fact_property(self) -> None:
        """Test user fact has fact property."""
        collection = SolaceCollections.user_fact()
        prop_names = [p.name for p in collection.properties]
        assert "fact" in prop_names
        assert "subject" in prop_names
        assert "predicate" in prop_names

    def test_crisis_event_collection(self) -> None:
        """Test crisis event collection definition."""
        collection = SolaceCollections.crisis_event()
        assert collection.name == "CrisisEvent"
        assert collection.collection_type == CollectionType.SAFETY

    def test_crisis_event_no_multi_tenancy(self) -> None:
        """Test crisis event has multi-tenancy disabled."""
        collection = SolaceCollections.crisis_event()
        assert collection.multi_tenancy is False

    def test_crisis_event_has_severity(self) -> None:
        """Test crisis event has severity property."""
        collection = SolaceCollections.crisis_event()
        prop_names = [p.name for p in collection.properties]
        assert "severity_level" in prop_names
        assert "event_type" in prop_names

    def test_all_collections_returns_list(self) -> None:
        """Test all_collections returns list of definitions."""
        collections = SolaceCollections.all_collections()
        assert isinstance(collections, list)
        assert len(collections) == 5

    def test_all_collections_includes_all_types(self) -> None:
        """Test all_collections includes all collection types."""
        collections = SolaceCollections.all_collections()
        names = [c.name for c in collections]
        assert "ConversationMemory" in names
        assert "SessionSummary" in names
        assert "TherapeuticInsight" in names
        assert "UserFact" in names
        assert "CrisisEvent" in names


class TestWeaviateSchemaManager:
    """Tests for WeaviateSchemaManager class."""

    def test_manager_initialization(self) -> None:
        """Test WeaviateSchemaManager can be initialized."""
        mock_client = MagicMock()
        manager = WeaviateSchemaManager(mock_client)
        assert manager is not None

    def test_manager_with_settings(self) -> None:
        """Test WeaviateSchemaManager with custom settings."""
        mock_client = MagicMock()
        settings = WeaviateSchemaSettings(replication_factor=3)
        manager = WeaviateSchemaManager(mock_client, settings)
        assert manager._settings.replication_factor == 3


class TestSetupWeaviateSchema:
    """Tests for setup_weaviate_schema function."""

    @pytest.mark.asyncio
    async def test_setup_returns_dict(self) -> None:
        """Test setup returns dictionary of results."""
        mock_client = MagicMock()
        mock_client.create_collection = AsyncMock(return_value=True)

        result = await setup_weaviate_schema(mock_client)
        assert isinstance(result, dict)
