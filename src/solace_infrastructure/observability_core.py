"""Solace-AI Observability - Structured logging, metrics, and distributed tracing."""
from __future__ import annotations
import asyncio
import functools
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, ParamSpec, TypeVar
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

P = ParamSpec("P")
R = TypeVar("R")

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_trace_context: ContextVar[dict[str, Any]] = ContextVar("trace_context", default={})


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ObservabilitySettings(BaseSettings):
    """Observability configuration from environment."""
    service_name: str = Field(default="solace-ai")
    environment: str = Field(default="development")
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: str = Field(default="json")
    enable_console: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    trace_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    metrics_prefix: str = Field(default="solace")
    otlp_endpoint: str | None = Field(default=None)
    otlp_headers: dict[str, str] = Field(default_factory=dict)
    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", env_file=".env", extra="ignore")


def get_correlation_id() -> str:
    """Get current correlation ID from context."""
    cid = _correlation_id.get()
    if not cid:
        cid = str(uuid.uuid4())
        _correlation_id.set(cid)
    return cid


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    _correlation_id.set(correlation_id)


def get_trace_context() -> dict[str, Any]:
    """Get current trace context."""
    return _trace_context.get()


def set_trace_context(context: dict[str, Any]) -> None:
    """Set trace context."""
    _trace_context.set(context)


def add_log_context(**kwargs: Any) -> structlog.BoundLogger:
    """Add context to the current logger."""
    return structlog.get_logger().bind(**kwargs)


def configure_logging(settings: ObservabilitySettings | None = None) -> None:
    """Configure structured logging with structlog."""
    settings = settings or ObservabilitySettings()
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_service_context(settings.service_name, settings.environment),
        _add_correlation_id,
    ]
    if settings.log_format == "json":
        processors.extend([
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ])
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    import logging
    log_level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            log_level_map.get(settings.log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _add_service_context(service_name: str, environment: str) -> Callable[..., Any]:
    """Processor to add service context to logs."""
    def processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        event_dict["service"] = service_name
        event_dict["environment"] = environment
        return event_dict
    return processor


def _add_correlation_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Processor to add correlation ID to logs."""
    cid = _correlation_id.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


@dataclass
class MetricValue:
    """Container for a metric value."""
    name: str
    value: float
    metric_type: MetricType
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsRegistry:
    """Simple in-memory metrics registry for collection and export."""

    def __init__(self, prefix: str = "solace") -> None:
        self._prefix = prefix
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._labels: dict[str, dict[str, str]] = {}

    def _make_key(self, name: str, labels: dict[str, str] | None = None) -> str:
        """Create unique key for metric with labels."""
        key = f"{self._prefix}_{name}"
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            key = f"{key}{{{label_str}}}"
        return key

    def counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
        if labels:
            self._labels[key] = labels

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        if labels:
            self._labels[key] = labels

    def histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        if labels:
            self._labels[key] = labels

    def get_all(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {k: self._histogram_stats(v) for k, v in self._histograms.items()},
        }

    def _histogram_stats(self, values: list[float]) -> dict[str, float]:
        """Calculate histogram statistics."""
        if not values:
            return {"count": 0, "sum": 0, "min": 0, "max": 0, "avg": 0}
        sorted_values = sorted(values)
        return {
            "count": len(values), "sum": sum(values),
            "min": min(values), "max": max(values),
            "avg": sum(values) / len(values),
            "p50": sorted_values[len(sorted_values) // 2],
            "p95": sorted_values[int(len(sorted_values) * 0.95)] if len(sorted_values) > 1 else sorted_values[0],
            "p99": sorted_values[int(len(sorted_values) * 0.99)] if len(sorted_values) > 1 else sorted_values[0],
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._labels.clear()


_metrics_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    """Get global metrics registry singleton."""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


@dataclass
class Span:
    """Represents a trace span for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({"name": name, "timestamp": time.perf_counter(), "attributes": attributes or {}})

    def end(self, status: str = "ok") -> None:
        self.end_time = time.perf_counter()
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id, "span_id": self.span_id, "parent_span_id": self.parent_span_id,
            "operation": self.operation_name, "duration_ms": round(self.duration_ms, 3),
            "attributes": self.attributes, "events": self.events, "status": self.status,
        }


class Tracer:
    """Simple distributed tracing implementation."""

    def __init__(self, service_name: str = "solace-ai", sample_rate: float = 1.0) -> None:
        self._service_name = service_name
        self._sample_rate = sample_rate
        self._active_spans: dict[str, Span] = {}
        self._completed_spans: list[Span] = []

    def start_span(self, operation_name: str, parent_span_id: str | None = None,
                   attributes: dict[str, Any] | None = None) -> Span:
        """Start a new span."""
        ctx = get_trace_context()
        trace_id = ctx.get("trace_id") or str(uuid.uuid4())[:16]
        span_id = str(uuid.uuid4())[:16]
        span = Span(
            trace_id=trace_id, span_id=span_id,
            parent_span_id=parent_span_id or ctx.get("span_id"),
            operation_name=operation_name, attributes=attributes or {},
        )
        span.set_attribute("service.name", self._service_name)
        self._active_spans[span_id] = span
        set_trace_context({"trace_id": trace_id, "span_id": span_id})
        return span

    def end_span(self, span: Span, status: str = "ok") -> None:
        """End a span and record it."""
        span.end(status)
        self._active_spans.pop(span.span_id, None)
        self._completed_spans.append(span)
        logger = structlog.get_logger()
        logger.debug("span_completed", **span.to_dict())

    def get_completed_spans(self, clear: bool = False) -> list[Span]:
        """Get completed spans."""
        spans = list(self._completed_spans)
        if clear:
            self._completed_spans.clear()
        return spans


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get global tracer singleton."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def traced(operation_name: str | None = None, record_args: bool = False) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to trace function execution."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        name = operation_name or func.__qualname__
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = get_tracer()
            attrs = {"function": func.__qualname__}
            if record_args:
                attrs["args_count"] = len(args)
                attrs["kwargs_keys"] = list(kwargs.keys())
            span = tracer.start_span(name, attributes=attrs)
            try:
                result = await func(*args, **kwargs)
                tracer.end_span(span, "ok")
                return result
            except Exception as e:
                span.set_attribute("error", str(e))
                tracer.end_span(span, "error")
                raise
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = get_tracer()
            attrs = {"function": func.__qualname__}
            span = tracer.start_span(name, attributes=attrs)
            try:
                result = func(*args, **kwargs)
                tracer.end_span(span, "ok")
                return result
            except Exception as e:
                span.set_attribute("error", str(e))
                tracer.end_span(span, "error")
                raise
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def timed(metric_name: str, labels: dict[str, str] | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to record function execution time as histogram."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                get_metrics_registry().histogram(metric_name, duration, labels)
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                get_metrics_registry().histogram(metric_name, duration, labels)
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def counted(metric_name: str, labels: dict[str, str] | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to count function invocations."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            get_metrics_registry().counter(metric_name, 1, labels)
            return await func(*args, **kwargs)
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            get_metrics_registry().counter(metric_name, 1, labels)
            return func(*args, **kwargs)
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
