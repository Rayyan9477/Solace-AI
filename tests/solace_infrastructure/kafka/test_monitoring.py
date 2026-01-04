"""Unit tests for Kafka Monitoring module."""
from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from solace_infrastructure.kafka.monitoring import (
    HealthStatus,
    AlertSeverity,
    BrokerMetrics,
    TopicMetrics,
    ConsumerGroupMetrics,
    ConsumerLag,
    KafkaAlert,
    ClusterHealth,
    MonitoringSettings,
    MetricsCollector,
    KafkaMonitorAdapter,
    KafkaMonitor,
    create_kafka_monitor,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self) -> None:
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self) -> None:
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestMonitoringSettings:
    """Tests for MonitoringSettings."""

    def test_default_settings(self) -> None:
        settings = MonitoringSettings()
        assert settings.bootstrap_servers == "localhost:9092"
        assert settings.poll_interval_seconds == 30
        assert settings.lag_threshold_warning == 1000
        assert settings.lag_threshold_critical == 10000

    def test_custom_settings(self) -> None:
        settings = MonitoringSettings(
            bootstrap_servers="kafka1:9092,kafka2:9092",
            poll_interval_seconds=60,
            lag_threshold_warning=500,
            enable_alerting=False,
        )
        assert "kafka1:9092" in settings.bootstrap_servers
        assert settings.poll_interval_seconds == 60
        assert settings.enable_alerting is False


class TestBrokerMetrics:
    """Tests for BrokerMetrics dataclass."""

    def test_broker_creation(self) -> None:
        broker = BrokerMetrics(
            broker_id=1,
            host="kafka-1",
            port=9092,
            is_controller=True,
        )
        assert broker.broker_id == 1
        assert broker.host == "kafka-1"
        assert broker.is_controller is True
        assert broker.is_alive is True

    def test_broker_with_rack(self) -> None:
        broker = BrokerMetrics(
            broker_id=2,
            host="kafka-2",
            port=9092,
            rack="us-east-1a",
        )
        assert broker.rack == "us-east-1a"


class TestTopicMetrics:
    """Tests for TopicMetrics dataclass."""

    def test_topic_metrics(self) -> None:
        metrics = TopicMetrics(
            topic_name="test.topic",
            partition_count=4,
            replication_factor=3,
            message_count=10000,
        )
        assert metrics.topic_name == "test.topic"
        assert metrics.partition_count == 4
        assert metrics.replication_factor == 3

    def test_topic_with_issues(self) -> None:
        metrics = TopicMetrics(
            topic_name="test.topic",
            partition_count=4,
            under_replicated=2,
            offline_partitions=1,
        )
        assert metrics.under_replicated == 2
        assert metrics.offline_partitions == 1


class TestConsumerGroupMetrics:
    """Tests for ConsumerGroupMetrics dataclass."""

    def test_group_metrics(self) -> None:
        metrics = ConsumerGroupMetrics(
            group_id="test-group",
            state="Stable",
            members_count=3,
            topics=["topic1", "topic2"],
            total_lag=100,
        )
        assert metrics.group_id == "test-group"
        assert metrics.state == "Stable"
        assert metrics.members_count == 3
        assert len(metrics.topics) == 2


class TestConsumerLag:
    """Tests for ConsumerLag dataclass."""

    def test_lag_creation(self) -> None:
        lag = ConsumerLag(
            topic="test.topic",
            partition=0,
            consumer_group="test-group",
            current_offset=100,
            end_offset=150,
            lag=50,
        )
        assert lag.topic == "test.topic"
        assert lag.partition == 0
        assert lag.lag == 50

    def test_lag_zero(self) -> None:
        lag = ConsumerLag(
            topic="test.topic",
            partition=0,
            consumer_group="test-group",
            current_offset=150,
            end_offset=150,
            lag=0,
        )
        assert lag.lag == 0


class TestKafkaAlert:
    """Tests for KafkaAlert dataclass."""

    def test_alert_creation(self) -> None:
        alert = KafkaAlert(
            alert_id="test-alert-1",
            severity=AlertSeverity.WARNING,
            title="High Consumer Lag",
            message="Consumer group lag exceeded threshold",
            source="kafka_monitor",
        )
        assert alert.alert_id == "test-alert-1"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.timestamp is not None

    def test_alert_with_metadata(self) -> None:
        alert = KafkaAlert(
            alert_id="test-alert-2",
            severity=AlertSeverity.CRITICAL,
            title="Broker Down",
            message="Broker 1 is unreachable",
            source="kafka_monitor",
            metadata={"broker_id": 1},
        )
        assert alert.metadata["broker_id"] == 1


class TestClusterHealth:
    """Tests for ClusterHealth dataclass."""

    def test_healthy_cluster(self) -> None:
        health = ClusterHealth(
            status=HealthStatus.HEALTHY,
            brokers_total=3,
            brokers_alive=3,
            topics_total=10,
            under_replicated_partitions=0,
            offline_partitions=0,
            total_consumer_lag=100,
        )
        assert health.status == HealthStatus.HEALTHY
        assert health.brokers_alive == health.brokers_total

    def test_degraded_cluster(self) -> None:
        health = ClusterHealth(
            status=HealthStatus.DEGRADED,
            brokers_total=3,
            brokers_alive=2,
            topics_total=10,
            under_replicated_partitions=5,
            offline_partitions=0,
            total_consumer_lag=5000,
        )
        assert health.status == HealthStatus.DEGRADED
        assert health.brokers_alive < health.brokers_total


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        return MetricsCollector(prefix="test")

    def test_gauge(self, collector: MetricsCollector) -> None:
        collector.gauge("connections", 10.0)
        metrics = collector.get_all()
        assert "test_connections" in metrics
        assert metrics["test_connections"] == 10.0

    def test_gauge_with_labels(self, collector: MetricsCollector) -> None:
        collector.gauge("broker_connections", 5.0, {"broker_id": "1"})
        metrics = collector.get_all()
        assert any("broker_connections" in k and "broker_id" in k for k in metrics)

    def test_increment(self, collector: MetricsCollector) -> None:
        collector.increment("requests")
        collector.increment("requests")
        collector.increment("requests", 3.0)
        metrics = collector.get_all()
        assert metrics["test_requests"] == 5.0

    def test_increment_with_labels(self, collector: MetricsCollector) -> None:
        collector.increment("messages", 100.0, {"topic": "test"})
        metrics = collector.get_all()
        assert any("messages" in k for k in metrics)

    def test_export_prometheus(self, collector: MetricsCollector) -> None:
        collector.gauge("active_connections", 5.0)
        collector.increment("total_messages", 100.0)
        output = collector.export_prometheus()
        assert "test_active_connections 5.0" in output
        assert "test_total_messages 100.0" in output


class TestKafkaMonitorAdapter:
    """Tests for KafkaMonitorAdapter."""

    @pytest.fixture
    def adapter(self) -> KafkaMonitorAdapter:
        settings = MonitoringSettings()
        return KafkaMonitorAdapter(settings)

    @pytest.mark.asyncio
    async def test_connect_close(self, adapter: KafkaMonitorAdapter) -> None:
        await adapter.connect()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_brokers_mock(self, adapter: KafkaMonitorAdapter) -> None:
        await adapter.connect()
        brokers = await adapter.get_brokers()
        assert len(brokers) >= 1
        assert brokers[0].is_alive is True
        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_topic_metrics_mock(self, adapter: KafkaMonitorAdapter) -> None:
        await adapter.connect()
        metrics = await adapter.get_topic_metrics("test.topic")
        assert metrics is not None
        assert metrics.partition_count == 4
        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_consumer_groups_mock(self, adapter: KafkaMonitorAdapter) -> None:
        await adapter.connect()
        groups = await adapter.get_consumer_groups()
        assert isinstance(groups, list)
        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_consumer_lag_mock(self, adapter: KafkaMonitorAdapter) -> None:
        await adapter.connect()
        lags = await adapter.get_consumer_lag("test-group")
        assert isinstance(lags, list)
        await adapter.close()


class TestKafkaMonitor:
    """Tests for KafkaMonitor."""

    @pytest.fixture
    def monitor(self) -> KafkaMonitor:
        settings = MonitoringSettings(poll_interval_seconds=60)
        return KafkaMonitor(settings)

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor: KafkaMonitor) -> None:
        await monitor.start()
        await asyncio.sleep(0.1)
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_get_cluster_health(self, monitor: KafkaMonitor) -> None:
        await monitor.start()
        health = await monitor.get_cluster_health()
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_get_metrics(self, monitor: KafkaMonitor) -> None:
        await monitor.start()
        await asyncio.sleep(0.1)
        metrics = monitor.get_metrics()
        assert isinstance(metrics, dict)
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_export_prometheus(self, monitor: KafkaMonitor) -> None:
        await monitor.start()
        prometheus_output = monitor.export_prometheus()
        assert isinstance(prometheus_output, str)
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_register_alert_callback(self, monitor: KafkaMonitor) -> None:
        alerts_received: list[KafkaAlert] = []

        async def alert_handler(alert: KafkaAlert) -> None:
            alerts_received.append(alert)

        monitor.register_alert_callback(alert_handler)
        # Alert callback is registered but we'd need to trigger one
        await monitor.start()
        await monitor.stop()


class TestFactoryFunction:
    """Tests for factory functions."""

    @pytest.mark.asyncio
    async def test_create_kafka_monitor(self) -> None:
        monitor = await create_kafka_monitor()
        assert isinstance(monitor, KafkaMonitor)
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_create_kafka_monitor_with_settings(self) -> None:
        settings = MonitoringSettings(
            poll_interval_seconds=120,
            lag_threshold_warning=500,
        )
        monitor = await create_kafka_monitor(settings)
        assert isinstance(monitor, KafkaMonitor)
        await monitor.stop()
