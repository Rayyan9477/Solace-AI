"""Unit tests for Kafka Topic Management module."""
from __future__ import annotations

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from solace_infrastructure.kafka.topics import (
    CleanupPolicy,
    CompressionCodec,
    TimestampType,
    TopicPriority,
    TopicDefinition,
    TopicAdminSettings,
    TopicMetadata,
    TopicOperationResult,
    TopicValidator,
    KafkaAdminAdapter,
    TopicManager,
    create_topic_manager,
)


class TestTopicDefinition:
    """Tests for TopicDefinition model."""

    def test_default_values(self) -> None:
        topic = TopicDefinition(name="test.topic")
        assert topic.partitions == 4
        assert topic.replication_factor == 3
        assert topic.cleanup_policy == CleanupPolicy.DELETE
        assert topic.min_insync_replicas == 2

    def test_custom_values(self) -> None:
        topic = TopicDefinition(
            name="custom.topic",
            partitions=8,
            replication_factor=3,
            cleanup_policy=CleanupPolicy.COMPACT,
            retention_ms=86400000,
        )
        assert topic.partitions == 8
        assert topic.cleanup_policy == CleanupPolicy.COMPACT
        assert topic.retention_ms == 86400000

    def test_name_validation_valid(self) -> None:
        valid_names = ["valid.topic", "valid-topic", "valid_topic", "ValidTopic123"]
        for name in valid_names:
            topic = TopicDefinition(name=name)
            assert topic.name == name.lower()

    def test_name_validation_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid topic name"):
            TopicDefinition(name="invalid@topic")

    def test_reserved_prefix_validation(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            TopicDefinition(name="__consumer_offsets")

    def test_to_kafka_config(self) -> None:
        topic = TopicDefinition(
            name="test.topic",
            retention_ms=604800000,
            cleanup_policy=CleanupPolicy.COMPACT_DELETE,
        )
        config = topic.to_kafka_config()
        assert config["retention.ms"] == "604800000"
        assert config["cleanup.policy"] == "compact,delete"
        assert config["min.insync.replicas"] == "2"

    def test_tags(self) -> None:
        topic = TopicDefinition(
            name="tagged.topic",
            tags={"env": "prod", "team": "platform"},
        )
        assert topic.tags["env"] == "prod"
        assert topic.tags["team"] == "platform"


class TestTopicAdminSettings:
    """Tests for TopicAdminSettings."""

    def test_default_settings(self) -> None:
        settings = TopicAdminSettings()
        assert settings.bootstrap_servers == "localhost:9092"
        assert settings.admin_client_id == "solace-admin"
        assert settings.request_timeout_ms == 30000

    def test_custom_settings(self) -> None:
        settings = TopicAdminSettings(
            bootstrap_servers="kafka1:9092,kafka2:9092",
            admin_client_id="custom-admin",
            default_replication_factor=3,
        )
        assert "kafka1:9092" in settings.bootstrap_servers
        assert settings.admin_client_id == "custom-admin"


class TestTopicValidator:
    """Tests for TopicValidator."""

    def test_validate_valid_name(self) -> None:
        valid, error = TopicValidator.validate_name("valid.topic.name")
        assert valid is True
        assert error is None

    def test_validate_empty_name(self) -> None:
        valid, error = TopicValidator.validate_name("")
        assert valid is False
        assert "empty" in error.lower()

    def test_validate_too_long_name(self) -> None:
        long_name = "a" * 250
        valid, error = TopicValidator.validate_name(long_name)
        assert valid is False
        assert "249" in error

    def test_validate_reserved_prefix(self) -> None:
        valid, error = TopicValidator.validate_name("__internal")
        assert valid is False
        assert "reserved" in error.lower()

    def test_validate_invalid_characters(self) -> None:
        valid, error = TopicValidator.validate_name("invalid!topic")
        assert valid is False
        assert "invalid characters" in error.lower()

    def test_validate_consecutive_dots(self) -> None:
        valid, error = TopicValidator.validate_name("invalid..topic")
        assert valid is False
        assert "dots" in error.lower()

    def test_validate_config_warnings(self) -> None:
        topic = TopicDefinition(name="test.topic", replication_factor=1)
        warnings = TopicValidator.validate_config(topic)
        assert any("replication" in w.lower() for w in warnings)

    def test_validate_config_min_isr_warning(self) -> None:
        topic = TopicDefinition(name="test.topic", replication_factor=2, min_insync_replicas=2)
        warnings = TopicValidator.validate_config(topic)
        assert any("min.insync.replicas" in w for w in warnings)

    def test_validate_config_infinite_retention(self) -> None:
        topic = TopicDefinition(name="test.topic", retention_ms=-1, retention_bytes=-1)
        warnings = TopicValidator.validate_config(topic)
        assert any("infinite" in w.lower() for w in warnings)

    def test_validate_config_high_partitions(self) -> None:
        topic = TopicDefinition(name="test.topic", partitions=100)
        warnings = TopicValidator.validate_config(topic)
        assert any("partition count" in w.lower() for w in warnings)


class TestKafkaAdminAdapter:
    """Tests for KafkaAdminAdapter."""

    @pytest.mark.asyncio
    async def test_connect_mock_mode(self) -> None:
        settings = TopicAdminSettings()
        adapter = KafkaAdminAdapter(settings)
        await adapter.connect()
        # Should be in mock mode without actual Kafka
        await adapter.close()

    @pytest.mark.asyncio
    async def test_create_topic_mock(self) -> None:
        settings = TopicAdminSettings()
        adapter = KafkaAdminAdapter(settings)
        await adapter.connect()
        topic = TopicDefinition(name="test.topic")
        result = await adapter.create_topic(topic)
        assert result.success is True
        assert result.operation == "create"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_delete_topic_mock(self) -> None:
        settings = TopicAdminSettings()
        adapter = KafkaAdminAdapter(settings)
        await adapter.connect()
        result = await adapter.delete_topic("test.topic")
        assert result.success is True
        assert result.operation == "delete"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_describe_topic_mock(self) -> None:
        settings = TopicAdminSettings()
        adapter = KafkaAdminAdapter(settings)
        await adapter.connect()
        metadata = await adapter.describe_topic("test.topic")
        assert metadata is not None
        assert metadata.partitions == 4
        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_topics_mock(self) -> None:
        settings = TopicAdminSettings()
        adapter = KafkaAdminAdapter(settings)
        await adapter.connect()
        topics = await adapter.list_topics()
        assert isinstance(topics, list)
        await adapter.close()


class TestTopicManager:
    """Tests for TopicManager."""

    @pytest.fixture
    def manager(self) -> TopicManager:
        return TopicManager()

    @pytest.mark.asyncio
    async def test_connect_close(self, manager: TopicManager) -> None:
        await manager.connect()
        await manager.close()

    def test_register_topic(self, manager: TopicManager) -> None:
        topic = TopicDefinition(name="registered.topic")
        manager.register_topic(topic)
        registered = manager.get_registered_topics()
        assert "registered.topic" in registered

    def test_register_solace_topics(self, manager: TopicManager) -> None:
        manager.register_solace_topics()
        registered = manager.get_registered_topics()
        assert "solace.sessions" in registered
        assert "solace.safety" in registered
        assert "solace.sessions.dlq" in registered

    @pytest.mark.asyncio
    async def test_create_topic(self, manager: TopicManager) -> None:
        await manager.connect()
        topic = TopicDefinition(name="new.topic")
        result = await manager.create_topic(topic)
        assert result.success is True
        await manager.close()

    @pytest.mark.asyncio
    async def test_create_topic_validation_failure(self, manager: TopicManager) -> None:
        await manager.connect()
        # Reserved prefix names are rejected at TopicDefinition validation level
        with pytest.raises(ValueError, match="reserved"):
            TopicDefinition(name="__reserved")
        await manager.close()

    @pytest.mark.asyncio
    async def test_ensure_topic(self, manager: TopicManager) -> None:
        await manager.connect()
        topic = TopicDefinition(name="ensured.topic")
        result = await manager.ensure_topic(topic)
        assert result.success is True
        await manager.close()

    @pytest.mark.asyncio
    async def test_delete_topic(self, manager: TopicManager) -> None:
        await manager.connect()
        result = await manager.delete_topic("test.topic")
        assert result.success is True
        await manager.close()

    @pytest.mark.asyncio
    async def test_list_all_topics(self, manager: TopicManager) -> None:
        await manager.connect()
        topics = await manager.list_all_topics()
        assert isinstance(topics, list)
        await manager.close()


class TestTopicOperationResult:
    """Tests for TopicOperationResult."""

    def test_success_result(self) -> None:
        result = TopicOperationResult(
            success=True,
            topic_name="test.topic",
            operation="create",
            message="Topic created successfully",
        )
        assert result.success is True
        assert result.error_code is None

    def test_failure_result(self) -> None:
        result = TopicOperationResult(
            success=False,
            topic_name="test.topic",
            operation="create",
            message="Already exists",
            error_code="TOPIC_EXISTS",
        )
        assert result.success is False
        assert result.error_code == "TOPIC_EXISTS"


class TestEnums:
    """Tests for topic-related enums."""

    def test_cleanup_policy_values(self) -> None:
        assert CleanupPolicy.DELETE.value == "delete"
        assert CleanupPolicy.COMPACT.value == "compact"
        assert CleanupPolicy.COMPACT_DELETE.value == "compact,delete"

    def test_compression_codec_values(self) -> None:
        assert CompressionCodec.GZIP.value == "gzip"
        assert CompressionCodec.SNAPPY.value == "snappy"
        assert CompressionCodec.LZ4.value == "lz4"

    def test_timestamp_type_values(self) -> None:
        assert TimestampType.CREATE_TIME.value == "CreateTime"
        assert TimestampType.LOG_APPEND_TIME.value == "LogAppendTime"

    def test_topic_priority_values(self) -> None:
        assert TopicPriority.CRITICAL.value == "critical"
        assert TopicPriority.HIGH.value == "high"
        assert TopicPriority.NORMAL.value == "normal"


class TestFactoryFunction:
    """Tests for factory functions."""

    @pytest.mark.asyncio
    async def test_create_topic_manager(self) -> None:
        manager = await create_topic_manager()
        assert isinstance(manager, TopicManager)
        await manager.close()
