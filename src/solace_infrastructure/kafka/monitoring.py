"""Solace-AI Kafka Monitoring - Metrics collection and health monitoring."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN = "healthy", "degraded", "unhealthy", "unknown"


class AlertSeverity(str, Enum):
    INFO, WARNING, ERROR, CRITICAL = "info", "warning", "error", "critical"


@dataclass
class BrokerMetrics:
    broker_id: int
    host: str
    port: int
    is_controller: bool = False
    is_alive: bool = True
    rack: str | None = None
    partition_count: int = 0
    leader_count: int = 0


@dataclass
class TopicMetrics:
    topic_name: str
    partition_count: int = 0
    replication_factor: int = 0
    message_count: int = 0
    size_bytes: int = 0
    under_replicated: int = 0
    offline_partitions: int = 0


@dataclass
class ConsumerGroupMetrics:
    group_id: str
    state: str = "Unknown"
    members_count: int = 0
    topics: list[str] = field(default_factory=list)
    total_lag: int = 0
    partition_lags: dict[str, int] = field(default_factory=dict)


@dataclass
class ConsumerLag:
    topic: str
    partition: int
    consumer_group: str
    current_offset: int
    end_offset: int
    lag: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class KafkaAlert:
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterHealth:
    status: HealthStatus
    brokers_total: int
    brokers_alive: int
    topics_total: int
    under_replicated_partitions: int
    offline_partitions: int
    total_consumer_lag: int
    alerts: list[KafkaAlert] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MonitoringSettings(BaseSettings):
    bootstrap_servers: str = Field(default="localhost:9092")
    poll_interval_seconds: int = Field(default=30, ge=5)
    lag_threshold_warning: int = Field(default=1000, ge=0)
    lag_threshold_critical: int = Field(default=10000, ge=0)
    broker_timeout_ms: int = Field(default=5000, ge=1000)
    enable_alerting: bool = Field(default=True)
    metrics_prefix: str = Field(default="solace_kafka")
    model_config = SettingsConfigDict(env_prefix="KAFKA_MONITOR_", env_file=".env", extra="ignore")


class MetricsCollector:
    def __init__(self, prefix: str = "solace_kafka") -> None:
        self._prefix = prefix
        self._metrics: dict[str, float] = {}
        self._labels: dict[str, dict[str, str]] = {}

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._metric_key(name, labels)
        self._metrics[key] = value
        if labels:
            self._labels[key] = labels

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._metric_key(name, labels)
        self._metrics[key] = self._metrics.get(key, 0.0) + value

    def _metric_key(self, name: str, labels: dict[str, str] | None) -> str:
        full_name = f"{self._prefix}_{name}"
        if labels:
            return f"{full_name}{{{','.join(f'{k}=\"{v}\"' for k, v in sorted(labels.items()))}}}"
        return full_name

    def get_all(self) -> dict[str, float]:
        return dict(self._metrics)

    def export_prometheus(self) -> str:
        return "\n".join(f"{key} {value}" for key, value in self._metrics.items())


class KafkaMonitorAdapter:
    def __init__(self, settings: MonitoringSettings) -> None:
        self._settings = settings
        self._client: Any = None

    async def connect(self) -> None:
        try:
            from aiokafka.admin import AIOKafkaAdminClient
            self._client = AIOKafkaAdminClient(
                bootstrap_servers=self._settings.bootstrap_servers,
                client_id="solace-monitor", request_timeout_ms=self._settings.broker_timeout_ms)
            await self._client.start()
            logger.info("kafka_monitor_connected", servers=self._settings.bootstrap_servers)
        except ImportError:
            logger.warning("aiokafka_not_available", fallback="mock_mode")
            self._client = None
        except Exception as e:
            logger.error("kafka_monitor_connection_failed", error=str(e))
            self._client = None

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("kafka_monitor_disconnected")

    async def get_brokers(self) -> list[BrokerMetrics]:
        if not self._client:
            return [BrokerMetrics(broker_id=1, host="localhost", port=9092, is_alive=True)]
        try:
            cluster = await self._client.describe_cluster()
            return [BrokerMetrics(broker_id=b.node_id, host=b.host, port=b.port, rack=b.rack,
                                  is_controller=b.node_id == cluster.controller_id, is_alive=True)
                    for b in cluster.brokers]
        except Exception as e:
            logger.error("broker_fetch_failed", error=str(e))
            return []

    async def get_topic_metrics(self, topic: str) -> TopicMetrics | None:
        if not self._client:
            return TopicMetrics(topic_name=topic, partition_count=4, replication_factor=3)
        try:
            topics = await self._client.describe_topics([topic])
            if not topics:
                return None
            tm = topics[0]
            return TopicMetrics(
                topic_name=tm.topic, partition_count=len(tm.partitions),
                replication_factor=len(tm.partitions[0].replicas) if tm.partitions else 0,
                under_replicated=sum(1 for p in tm.partitions if len(p.isr) < len(p.replicas)),
                offline_partitions=sum(1 for p in tm.partitions if p.leader == -1))
        except Exception as e:
            logger.error("topic_metrics_failed", topic=topic, error=str(e))
            return None

    async def get_consumer_groups(self) -> list[ConsumerGroupMetrics]:
        if not self._client:
            return [ConsumerGroupMetrics(group_id="solace-group-analytics", state="Stable", members_count=2)]
        try:
            groups = await self._client.list_consumer_groups()
            return [ConsumerGroupMetrics(group_id=g.group_id if hasattr(g, "group_id") else str(g)) for g in groups]
        except Exception as e:
            logger.error("consumer_groups_failed", error=str(e))
            return []

    async def get_consumer_lag(self, group_id: str) -> list[ConsumerLag]:
        if not self._client:
            return [ConsumerLag(topic="solace.sessions", partition=0, consumer_group=group_id,
                               current_offset=100, end_offset=105, lag=5)]
        try:
            offsets = await self._client.list_consumer_group_offsets(group_id)
            lags: list[ConsumerLag] = []
            for tp, offset_meta in offsets.items():
                end_offset = (await self._client.end_offsets([tp])).get(tp, 0)
                lags.append(ConsumerLag(topic=tp.topic, partition=tp.partition, consumer_group=group_id,
                                       current_offset=offset_meta.offset, end_offset=end_offset,
                                       lag=max(0, end_offset - offset_meta.offset)))
            return lags
        except Exception as e:
            logger.error("consumer_lag_failed", group=group_id, error=str(e))
            return []


AlertCallback = Callable[[KafkaAlert], Awaitable[None]]


class KafkaMonitor:
    def __init__(self, settings: MonitoringSettings | None = None) -> None:
        self._settings = settings or MonitoringSettings()
        self._adapter = KafkaMonitorAdapter(self._settings)
        self._collector = MetricsCollector(self._settings.metrics_prefix)
        self._alert_callbacks: list[AlertCallback] = []
        self._monitoring_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        await self._adapter.connect()
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("kafka_monitoring_started", interval=self._settings.poll_interval_seconds)

    async def stop(self) -> None:
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        await self._adapter.close()
        logger.info("kafka_monitoring_stopped")

    def register_alert_callback(self, callback: AlertCallback) -> None:
        self._alert_callbacks.append(callback)

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self._settings.poll_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("monitoring_error", error=str(e))
                await asyncio.sleep(5)

    async def _collect_metrics(self) -> None:
        brokers = await self._adapter.get_brokers()
        alive_count = sum(1 for b in brokers if b.is_alive)
        self._collector.gauge("brokers_total", len(brokers))
        self._collector.gauge("brokers_alive", alive_count)
        for b in brokers:
            self._collector.gauge("broker_is_alive", 1.0 if b.is_alive else 0.0, {"broker_id": str(b.broker_id)})
        groups = await self._adapter.get_consumer_groups()
        self._collector.gauge("consumer_groups_total", len(groups))
        for group in groups:
            lags = await self._adapter.get_consumer_lag(group.group_id)
            total_lag = sum(l.lag for l in lags)
            self._collector.gauge("consumer_lag_total", total_lag, {"group": group.group_id})
            if total_lag > self._settings.lag_threshold_critical:
                await self._emit_alert(AlertSeverity.CRITICAL, "High Consumer Lag",
                                       f"Group {group.group_id} lag: {total_lag}", {"group_id": group.group_id, "lag": total_lag})
            elif total_lag > self._settings.lag_threshold_warning:
                await self._emit_alert(AlertSeverity.WARNING, "Elevated Consumer Lag",
                                       f"Group {group.group_id} lag: {total_lag}", {"group_id": group.group_id, "lag": total_lag})

    async def _emit_alert(self, severity: AlertSeverity, title: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        if not self._settings.enable_alerting:
            return
        alert = KafkaAlert(alert_id=f"{title.lower().replace(' ', '_')}_{datetime.now(timezone.utc).timestamp()}",
                          severity=severity, title=title, message=message, source="kafka_monitor", metadata=metadata or {})
        logger.warning("kafka_alert", severity=severity.value, title=title, message=message)
        for callback in self._alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error("alert_callback_failed", error=str(e))

    async def get_cluster_health(self) -> ClusterHealth:
        brokers = await self._adapter.get_brokers()
        alive_count = sum(1 for b in brokers if b.is_alive)
        total_lag = 0
        for g in await self._adapter.get_consumer_groups():
            total_lag += sum(l.lag for l in await self._adapter.get_consumer_lag(g.group_id))
        status = HealthStatus.UNHEALTHY if alive_count == 0 else (HealthStatus.DEGRADED if alive_count < len(brokers) else HealthStatus.HEALTHY)
        return ClusterHealth(status=status, brokers_total=len(brokers), brokers_alive=alive_count, topics_total=0,
                            under_replicated_partitions=0, offline_partitions=0, total_consumer_lag=total_lag)

    def get_metrics(self) -> dict[str, float]:
        return self._collector.get_all()

    def export_prometheus(self) -> str:
        return self._collector.export_prometheus()


async def create_kafka_monitor(settings: MonitoringSettings | None = None) -> KafkaMonitor:
    monitor = KafkaMonitor(settings)
    await monitor.start()
    return monitor
