"""Unit tests for Solace-AI Events Configuration."""

import pytest

from solace_events.src.config import (
    CompressionType,
    ConsumerGroup,
    ConsumerSettings,
    KafkaSettings,
    ProducerSettings,
    SaslMechanism,
    SecurityProtocol,
    SolaceTopic,
    TopicConfig,
    TOPIC_CONFIGS,
    get_all_topics,
    get_dlq_topics,
    get_topic_config,
)


class TestTopicConfig:
    """Tests for TopicConfig."""

    def test_default_values(self) -> None:
        """Test default topic configuration."""
        config = TopicConfig(name="test.topic")

        assert config.name == "test.topic"
        assert config.partitions == 4
        assert config.replication_factor == 3
        assert config.cleanup_policy == "delete"

    def test_custom_values(self) -> None:
        """Test custom topic configuration."""
        config = TopicConfig(
            name="custom.topic",
            partitions=8,
            replication_factor=2,
            retention_ms=86400000,
        )

        assert config.partitions == 8
        assert config.replication_factor == 2
        assert config.retention_ms == 86400000

    def test_name_validation(self) -> None:
        """Test topic name validation."""
        config = TopicConfig(name="Valid.Topic-Name_123")
        assert config.name == "valid.topic-name_123"

    def test_invalid_name(self) -> None:
        """Test invalid topic name rejected."""
        with pytest.raises(ValueError):
            TopicConfig(name="invalid@topic!")

    def test_immutability(self) -> None:
        """Test config is frozen."""
        config = TopicConfig(name="test.topic")
        with pytest.raises(Exception):
            config.partitions = 10  # type: ignore[misc]


class TestSolaceTopic:
    """Tests for SolaceTopic enum."""

    def test_all_topics_defined(self) -> None:
        """Test all expected topics are defined."""
        assert SolaceTopic.SESSIONS.value == "solace.sessions"
        assert SolaceTopic.ASSESSMENTS.value == "solace.assessments"
        assert SolaceTopic.THERAPY.value == "solace.therapy"
        assert SolaceTopic.SAFETY.value == "solace.safety"
        assert SolaceTopic.MEMORY.value == "solace.memory"
        assert SolaceTopic.ANALYTICS.value == "solace.analytics"
        assert SolaceTopic.PERSONALITY.value == "solace.personality"

    def test_dlq_topic(self) -> None:
        """Test DLQ topic name generation."""
        assert SolaceTopic.SAFETY.dlq_topic == "solace.safety.dlq"
        assert SolaceTopic.MEMORY.dlq_topic == "solace.memory.dlq"

    def test_from_string(self) -> None:
        """Test creating topic from string."""
        topic = SolaceTopic.from_string("solace.safety")
        assert topic == SolaceTopic.SAFETY

    def test_from_string_invalid(self) -> None:
        """Test invalid string raises error."""
        with pytest.raises(ValueError):
            SolaceTopic.from_string("invalid.topic")


class TestKafkaSettings:
    """Tests for KafkaSettings."""

    def test_default_values(self) -> None:
        """Test default settings."""
        settings = KafkaSettings()

        assert settings.bootstrap_servers == "localhost:9092"
        assert settings.security_protocol == SecurityProtocol.PLAINTEXT
        assert settings.client_id == "solace-ai"

    def test_connection_params_plaintext(self) -> None:
        """Test connection params for plaintext."""
        settings = KafkaSettings()
        params = settings.get_connection_params()

        assert params["bootstrap_servers"] == "localhost:9092"
        assert params["client_id"] == "solace-ai"
        assert "security_protocol" not in params

    def test_connection_params_sasl(self) -> None:
        """Test connection params for SASL auth."""
        settings = KafkaSettings(
            security_protocol=SecurityProtocol.SASL_PLAINTEXT,
            sasl_mechanism=SaslMechanism.PLAIN,
            sasl_username="user",
            sasl_password="pass",
        )
        params = settings.get_connection_params()

        assert params["security_protocol"] == "SASL_PLAINTEXT"
        assert params["sasl_mechanism"] == "PLAIN"
        assert params["sasl_plain_username"] == "user"


class TestProducerSettings:
    """Tests for ProducerSettings."""

    def test_default_values(self) -> None:
        """Test default producer settings."""
        settings = ProducerSettings()

        assert settings.acks == "all"
        assert settings.compression_type == CompressionType.GZIP
        assert settings.enable_idempotence is True

    def test_to_producer_params(self) -> None:
        """Test conversion to aiokafka params."""
        settings = ProducerSettings()
        params = settings.to_producer_params()

        assert params["acks"] == "all"
        assert params["compression_type"] == "gzip"
        assert params["enable_idempotence"] is True


class TestConsumerSettings:
    """Tests for ConsumerSettings."""

    def test_required_group_id(self) -> None:
        """Test group_id is required."""
        settings = ConsumerSettings(group_id="test-group")
        assert settings.group_id == "test-group"

    def test_default_values(self) -> None:
        """Test default consumer settings."""
        settings = ConsumerSettings(group_id="test-group")

        assert settings.auto_offset_reset == "earliest"
        assert settings.enable_auto_commit is False
        assert settings.max_poll_records == 100

    def test_group_id_validation(self) -> None:
        """Test group ID validation."""
        settings = ConsumerSettings(group_id="valid-group_id")
        assert settings.group_id == "valid-group_id"

    def test_invalid_group_id(self) -> None:
        """Test invalid group ID rejected."""
        with pytest.raises(ValueError):
            ConsumerSettings(group_id="invalid@group!")

    def test_to_consumer_params(self) -> None:
        """Test conversion to aiokafka params."""
        settings = ConsumerSettings(group_id="my-group")
        params = settings.to_consumer_params()

        assert params["group_id"] == "my-group"
        assert params["auto_offset_reset"] == "earliest"
        assert params["enable_auto_commit"] is False


class TestConsumerGroup:
    """Tests for ConsumerGroup enum."""

    def test_all_groups_defined(self) -> None:
        """Test all consumer groups are defined."""
        assert ConsumerGroup.ANALYTICS.value == "solace-group-analytics"
        assert ConsumerGroup.NOTIFICATIONS.value == "solace-group-notifications"
        assert ConsumerGroup.AUDIT.value == "solace-group-audit"
        assert ConsumerGroup.SAFETY_MONITOR.value == "solace-group-safety-monitor"


class TestTopicHelpers:
    """Tests for topic helper functions."""

    def test_get_topic_config(self) -> None:
        """Test getting topic config by enum."""
        config = get_topic_config(SolaceTopic.SAFETY)
        assert config.name == "solace.safety"
        assert config.partitions == 4

    def test_get_topic_config_by_string(self) -> None:
        """Test getting topic config by string."""
        config = get_topic_config("solace.safety")
        assert config.name == "solace.safety"

    def test_get_all_topics(self) -> None:
        """Test getting all topic names."""
        topics = get_all_topics()

        assert "solace.sessions" in topics
        assert "solace.safety" in topics
        assert len(topics) == 7

    def test_get_dlq_topics(self) -> None:
        """Test getting all DLQ topic names."""
        dlq_topics = get_dlq_topics()

        assert "solace.sessions.dlq" in dlq_topics
        assert "solace.safety.dlq" in dlq_topics
        assert len(dlq_topics) == 7

    def test_topic_configs_exist(self) -> None:
        """Test all topics have configs."""
        for topic in SolaceTopic:
            assert topic in TOPIC_CONFIGS
