"""
Solace-AI Analytics Service - ClickHouse Repository.
Implements Repository Pattern with async support for time-series analytics.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog

from models import TableName, AnalyticsEvent, MetricRecord, AggregationRecord

logger = structlog.get_logger(__name__)

_NIL_UUID = UUID(int=0)


def _safe_uuid(value: Any, field_name: str = "unknown") -> UUID:
    """Parse UUID safely, returning nil UUID on failure."""
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        logger.warning("invalid_uuid_in_row", field=field_name, value=str(value)[:50])
        return _NIL_UUID


class RepositoryError(Exception):
    """Base exception for repository errors."""

class RepositoryConnectionError(RepositoryError):
    """Raised when connection to storage fails."""

class QueryError(RepositoryError):
    """Raised when a query fails."""


@dataclass(frozen=True)
class ClickHouseConfig:
    """ClickHouse connection configuration."""
    host: str = "localhost"
    port: int = 8123
    database: str = "solace_analytics"
    username: str = "default"
    password: str = ""
    secure: bool = False
    verify: bool = True
    connect_timeout: float = 10.0
    query_timeout: float = 300.0
    max_connections: int = 10


class AnalyticsRepository(ABC):
    """Abstract base class for analytics data repository."""

    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
    @abstractmethod
    async def insert_event(self, event: AnalyticsEvent) -> None: ...
    @abstractmethod
    async def insert_events_batch(self, events: list[AnalyticsEvent]) -> int: ...
    @abstractmethod
    async def insert_metric(self, metric: MetricRecord) -> None: ...
    @abstractmethod
    async def insert_metrics_batch(self, metrics: list[MetricRecord]) -> int: ...
    @abstractmethod
    async def query_events(
        self, start_time: datetime, end_time: datetime,
        event_type: str | None = None, user_id: UUID | None = None, limit: int = 1000,
    ) -> list[AnalyticsEvent]: ...
    @abstractmethod
    async def query_metrics(
        self, metric_name: str, start_time: datetime, end_time: datetime,
        labels: dict[str, str] | None = None, limit: int = 1000,
    ) -> list[MetricRecord]: ...
    @abstractmethod
    async def get_aggregations(
        self, metric_name: str, window_type: str, start_time: datetime, end_time: datetime,
    ) -> list[AggregationRecord]: ...
    @abstractmethod
    async def store_aggregation(self, aggregation: AggregationRecord) -> None: ...


class ClickHouseRepository(AnalyticsRepository):
    """ClickHouse implementation of analytics repository."""

    def __init__(self, config: ClickHouseConfig) -> None:
        self._config = config
        self._client: Any = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lock:
            if self._connected:
                return
            try:
                import clickhouse_connect
                self._client = await asyncio.to_thread(
                    clickhouse_connect.get_client,
                    host=self._config.host, port=self._config.port,
                    database=self._config.database, username=self._config.username,
                    password=self._config.password, secure=self._config.secure,
                    verify=self._config.verify, connect_timeout=self._config.connect_timeout,
                    query_limit=0,
                )
                self._connected = True
                logger.info("clickhouse_connected", host=self._config.host)
            except ImportError:
                raise RepositoryConnectionError("clickhouse-connect package not installed")
            except Exception as e:
                logger.error("clickhouse_connection_failed", error=str(e))
                raise RepositoryConnectionError(f"Failed to connect to ClickHouse: {e}")

    async def disconnect(self) -> None:
        async with self._lock:
            if self._client:
                await asyncio.to_thread(self._client.close)
                self._client = None
                self._connected = False
                logger.info("clickhouse_disconnected")

    async def health_check(self) -> bool:
        if not self._connected or not self._client:
            return False
        try:
            return await asyncio.to_thread(self._client.command, "SELECT 1") == 1
        except Exception as e:
            logger.warning("clickhouse_health_check_failed", error=str(e))
            return False

    async def insert_event(self, event: AnalyticsEvent) -> None:
        await self.insert_events_batch([event])

    async def insert_events_batch(self, events: list[AnalyticsEvent]) -> int:
        if not self._connected:
            raise RepositoryConnectionError("Not connected to ClickHouse")
        if not events:
            return 0
        try:
            data = [[str(e.event_id), e.event_type, e.category, str(e.user_id),
                     str(e.session_id) if e.session_id else None, e.timestamp,
                     str(e.correlation_id), e.source_service, e.payload, e.created_at]
                    for e in events]
            await asyncio.to_thread(self._client.insert, TableName.EVENTS.value, data, column_names=[
                "event_id", "event_type", "category", "user_id", "session_id",
                "timestamp", "correlation_id", "source_service", "payload", "created_at"])
            logger.debug("events_inserted", count=len(events))
            return len(events)
        except Exception as e:
            logger.error("event_insert_failed", error=str(e))
            raise QueryError(f"Failed to insert events: {e}")

    async def insert_metric(self, metric: MetricRecord) -> None:
        await self.insert_metrics_batch([metric])

    async def insert_metrics_batch(self, metrics: list[MetricRecord]) -> int:
        if not self._connected:
            raise RepositoryConnectionError("Not connected to ClickHouse")
        if not metrics:
            return 0
        try:
            data = [[str(m.metric_id), m.metric_name, float(m.value), m.labels,
                     m.timestamp, m.window_start, m.window_end, m.window_type, m.created_at]
                    for m in metrics]
            await asyncio.to_thread(self._client.insert, TableName.METRICS.value, data, column_names=[
                "metric_id", "metric_name", "value", "labels", "timestamp",
                "window_start", "window_end", "window_type", "created_at"])
            logger.debug("metrics_inserted", count=len(metrics))
            return len(metrics)
        except Exception as e:
            logger.error("metric_insert_failed", error=str(e))
            raise QueryError(f"Failed to insert metrics: {e}")

    async def query_events(
        self, start_time: datetime, end_time: datetime,
        event_type: str | None = None, user_id: UUID | None = None, limit: int = 1000,
    ) -> list[AnalyticsEvent]:
        if not self._connected:
            raise RepositoryConnectionError("Not connected to ClickHouse")
        try:
            query = f"""SELECT event_id, event_type, category, user_id, session_id,
                timestamp, correlation_id, source_service, payload, created_at
                FROM {TableName.EVENTS.value}
                WHERE timestamp >= %(start_time)s AND timestamp < %(end_time)s"""
            params: dict[str, Any] = {"start_time": start_time, "end_time": end_time}
            if event_type:
                query += " AND event_type = %(event_type)s"
                params["event_type"] = event_type
            if user_id:
                query += " AND user_id = %(user_id)s"
                params["user_id"] = str(user_id)
            query += " ORDER BY timestamp DESC LIMIT %(limit)s"
            params["limit"] = int(limit)
            result = await asyncio.to_thread(self._client.query, query, parameters=params)
            return [AnalyticsEvent(
                event_id=_safe_uuid(row[0], "event_id"), event_type=row[1], category=row[2],
                user_id=_safe_uuid(row[3], "user_id"),
                session_id=_safe_uuid(row[4], "session_id") if row[4] else None,
                timestamp=row[5], correlation_id=_safe_uuid(row[6], "correlation_id"),
                source_service=row[7], payload=row[8] or {}, created_at=row[9],
            ) for row in result.result_rows]
        except Exception as e:
            logger.error("event_query_failed", error=str(e))
            raise QueryError(f"Failed to query events: {e}")

    async def query_metrics(
        self, metric_name: str, start_time: datetime, end_time: datetime,
        labels: dict[str, str] | None = None, limit: int = 1000,
    ) -> list[MetricRecord]:
        if not self._connected:
            raise RepositoryConnectionError("Not connected to ClickHouse")
        try:
            query = f"""SELECT metric_id, metric_name, value, labels, timestamp,
                window_start, window_end, window_type, created_at
                FROM {TableName.METRICS.value}
                WHERE metric_name = %(metric_name)s
                AND timestamp >= %(start_time)s AND timestamp < %(end_time)s
                ORDER BY timestamp DESC LIMIT %(limit)s"""
            params: dict[str, Any] = {
                "metric_name": metric_name, "start_time": start_time, "end_time": end_time,
                "limit": int(limit)}
            result = await asyncio.to_thread(self._client.query, query, parameters=params)
            return [MetricRecord(
                metric_id=_safe_uuid(row[0], "metric_id"), metric_name=row[1],
                value=Decimal(str(row[2])),
                labels=row[3] or {}, timestamp=row[4], window_start=row[5],
                window_end=row[6], window_type=row[7], created_at=row[8],
            ) for row in result.result_rows]
        except Exception as e:
            logger.error("metric_query_failed", error=str(e))
            raise QueryError(f"Failed to query metrics: {e}")

    async def get_aggregations(
        self, metric_name: str, window_type: str, start_time: datetime, end_time: datetime,
    ) -> list[AggregationRecord]:
        if not self._connected:
            raise RepositoryConnectionError("Not connected to ClickHouse")
        try:
            query = f"""SELECT aggregation_id, metric_name, window_type, window_start,
                window_end, count, sum_value, min_value, max_value, avg_value, labels, computed_at
                FROM {TableName.AGGREGATIONS.value}
                WHERE metric_name = %(metric_name)s AND window_type = %(window_type)s
                AND window_start >= %(start_time)s AND window_end <= %(end_time)s
                ORDER BY window_start"""
            params = {"metric_name": metric_name, "window_type": window_type,
                      "start_time": start_time, "end_time": end_time}
            result = await asyncio.to_thread(self._client.query, query, parameters=params)
            return [AggregationRecord(
                aggregation_id=_safe_uuid(row[0], "aggregation_id"), metric_name=row[1], window_type=row[2],
                window_start=row[3], window_end=row[4], count=row[5],
                sum_value=Decimal(str(row[6])),
                min_value=Decimal(str(row[7])) if row[7] else None,
                max_value=Decimal(str(row[8])) if row[8] else None,
                avg_value=Decimal(str(row[9])) if row[9] else None,
                labels=row[10] or {}, computed_at=row[11],
            ) for row in result.result_rows]
        except Exception as e:
            logger.error("aggregation_query_failed", error=str(e))
            raise QueryError(f"Failed to query aggregations: {e}")

    async def store_aggregation(self, aggregation: AggregationRecord) -> None:
        if not self._connected:
            raise RepositoryConnectionError("Not connected to ClickHouse")
        try:
            data = [[str(aggregation.aggregation_id), aggregation.metric_name,
                aggregation.window_type, aggregation.window_start, aggregation.window_end,
                aggregation.count, float(aggregation.sum_value),
                float(aggregation.min_value) if aggregation.min_value else None,
                float(aggregation.max_value) if aggregation.max_value else None,
                float(aggregation.avg_value) if aggregation.avg_value else None,
                aggregation.labels, aggregation.computed_at]]
            await asyncio.to_thread(self._client.insert, TableName.AGGREGATIONS.value, data, column_names=[
                "aggregation_id", "metric_name", "window_type", "window_start",
                "window_end", "count", "sum_value", "min_value", "max_value",
                "avg_value", "labels", "computed_at"])
        except Exception as e:
            logger.error("aggregation_store_failed", error=str(e))
            raise QueryError(f"Failed to store aggregation: {e}")


class InMemoryRepository(AnalyticsRepository):
    """In-memory implementation for testing and development."""

    def __init__(self) -> None:
        self._events: list[AnalyticsEvent] = []
        self._metrics: list[MetricRecord] = []
        self._aggregations: list[AggregationRecord] = []
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self._connected = True
        logger.info("inmemory_repository_connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("inmemory_repository_disconnected")

    async def health_check(self) -> bool:
        return self._connected

    async def insert_event(self, event: AnalyticsEvent) -> None:
        async with self._lock:
            self._events.append(event)

    async def insert_events_batch(self, events: list[AnalyticsEvent]) -> int:
        async with self._lock:
            self._events.extend(events)
            return len(events)

    async def insert_metric(self, metric: MetricRecord) -> None:
        async with self._lock:
            self._metrics.append(metric)

    async def insert_metrics_batch(self, metrics: list[MetricRecord]) -> int:
        async with self._lock:
            self._metrics.extend(metrics)
            return len(metrics)

    async def query_events(
        self, start_time: datetime, end_time: datetime,
        event_type: str | None = None, user_id: UUID | None = None, limit: int = 1000,
    ) -> list[AnalyticsEvent]:
        results = [e for e in self._events
                   if start_time <= e.timestamp < end_time
                   and (event_type is None or e.event_type == event_type)
                   and (user_id is None or e.user_id == user_id)]
        return sorted(results, key=lambda e: e.timestamp, reverse=True)[:limit]

    async def query_metrics(
        self, metric_name: str, start_time: datetime, end_time: datetime,
        labels: dict[str, str] | None = None, limit: int = 1000,
    ) -> list[MetricRecord]:
        results = [m for m in self._metrics
                   if m.metric_name == metric_name and start_time <= m.timestamp < end_time]
        return sorted(results, key=lambda m: m.timestamp, reverse=True)[:limit]

    async def get_aggregations(
        self, metric_name: str, window_type: str, start_time: datetime, end_time: datetime,
    ) -> list[AggregationRecord]:
        return [a for a in self._aggregations
                if a.metric_name == metric_name and a.window_type == window_type
                and a.window_start >= start_time and a.window_end <= end_time]

    async def store_aggregation(self, aggregation: AggregationRecord) -> None:
        async with self._lock:
            self._aggregations.append(aggregation)


_repository_instance: AnalyticsRepository | None = None
_repository_config: ClickHouseConfig | None = None


def configure_repository(config: ClickHouseConfig | None = None) -> None:
    """Configure the repository with ClickHouse settings."""
    global _repository_instance, _repository_config
    _repository_config = config
    _repository_instance = None  # Reset to pick up new config


def create_repository(config: ClickHouseConfig | None = None) -> AnalyticsRepository:
    """Factory function to create appropriate repository (non-singleton)."""
    return ClickHouseRepository(config) if config else InMemoryRepository()


def get_repository() -> AnalyticsRepository:
    """Get singleton repository instance.

    Uses ClickHouse when configured, otherwise uses in-memory.
    """
    global _repository_instance
    if _repository_instance is None:
        if _repository_config is not None:
            _repository_instance = ClickHouseRepository(_repository_config)
            logger.info("analytics_repository_created", type="clickhouse")
        else:
            _repository_instance = InMemoryRepository()
            logger.info("analytics_repository_created", type="in_memory")
    return _repository_instance


def reset_repository() -> None:
    """Reset the singleton repository (for testing)."""
    global _repository_instance, _repository_config
    _repository_instance = None
    _repository_config = None
