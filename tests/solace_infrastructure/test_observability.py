"""Unit tests for observability module."""
from __future__ import annotations
import asyncio
import time
from unittest.mock import patch, MagicMock
import pytest
from solace_infrastructure.observability import (
    ObservabilitySettings,
    LogLevel,
    MetricType,
    MetricValue,
    MetricsRegistry,
    Span,
    Tracer,
    configure_logging,
    get_correlation_id,
    set_correlation_id,
    get_trace_context,
    set_trace_context,
    add_log_context,
    get_metrics_registry,
    get_tracer,
    traced,
    timed,
    counted,
    _correlation_id,
)


class TestObservabilitySettings:
    """Tests for ObservabilitySettings configuration."""

    def test_default_settings(self):
        settings = ObservabilitySettings()
        assert settings.service_name == "solace-ai"
        assert settings.environment == "development"
        assert settings.log_level == LogLevel.INFO
        assert settings.log_format == "json"

    def test_custom_settings(self):
        settings = ObservabilitySettings(
            service_name="custom-service",
            environment="production",
            log_level=LogLevel.DEBUG,
            enable_tracing=False
        )
        assert settings.service_name == "custom-service"
        assert settings.environment == "production"
        assert settings.log_level == LogLevel.DEBUG
        assert not settings.enable_tracing


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_levels(self):
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_types(self):
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


class TestCorrelationId:
    """Tests for correlation ID context management."""

    def test_get_correlation_id_generates_new(self):
        _correlation_id.set("")
        cid = get_correlation_id()
        assert cid
        assert len(cid) == 36

    def test_set_and_get_correlation_id(self):
        test_id = "test-correlation-id"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id

    def test_correlation_id_persistence(self):
        test_id = "persistent-id"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id
        assert get_correlation_id() == test_id


class TestTraceContext:
    """Tests for trace context management."""

    def test_default_trace_context(self):
        set_trace_context({})
        ctx = get_trace_context()
        assert ctx == {}

    def test_set_trace_context(self):
        ctx = {"trace_id": "abc123", "span_id": "def456"}
        set_trace_context(ctx)
        result = get_trace_context()
        assert result["trace_id"] == "abc123"
        assert result["span_id"] == "def456"


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_metric_value_creation(self):
        metric = MetricValue(
            name="http_requests",
            value=42.0,
            metric_type=MetricType.COUNTER,
            labels={"method": "GET", "path": "/api/v1"}
        )
        assert metric.name == "http_requests"
        assert metric.value == 42.0
        assert metric.labels["method"] == "GET"


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    @pytest.fixture
    def registry(self):
        reg = MetricsRegistry(prefix="test")
        yield reg
        reg.reset()

    def test_counter(self, registry):
        registry.counter("requests")
        registry.counter("requests")
        registry.counter("requests", 3)
        metrics = registry.get_all()
        assert metrics["counters"]["test_requests"] == 5

    def test_counter_with_labels(self, registry):
        registry.counter("requests", labels={"method": "GET"})
        registry.counter("requests", labels={"method": "POST"})
        metrics = registry.get_all()
        assert "test_requests{method=GET}" in metrics["counters"]
        assert "test_requests{method=POST}" in metrics["counters"]

    def test_gauge(self, registry):
        registry.gauge("temperature", 25.5)
        registry.gauge("temperature", 26.0)
        metrics = registry.get_all()
        assert metrics["gauges"]["test_temperature"] == 26.0

    def test_histogram(self, registry):
        for i in range(10):
            registry.histogram("latency", i * 10)
        metrics = registry.get_all()
        stats = metrics["histograms"]["test_latency"]
        assert stats["count"] == 10
        assert stats["min"] == 0
        assert stats["max"] == 90

    def test_histogram_percentiles(self, registry):
        for i in range(100):
            registry.histogram("response_time", i)
        metrics = registry.get_all()
        stats = metrics["histograms"]["test_response_time"]
        assert stats["p50"] == 50
        assert stats["p95"] >= 94

    def test_reset(self, registry):
        registry.counter("test")
        registry.gauge("test", 1)
        registry.histogram("test", 1)
        registry.reset()
        metrics = registry.get_all()
        assert len(metrics["counters"]) == 0
        assert len(metrics["gauges"]) == 0
        assert len(metrics["histograms"]) == 0


class TestSpan:
    """Tests for Span tracing."""

    def test_span_creation(self):
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            operation_name="test_operation"
        )
        assert span.trace_id == "trace123"
        assert span.span_id == "span456"
        assert span.operation_name == "test_operation"
        assert span.status == "ok"

    def test_span_duration(self):
        span = Span(
            trace_id="t1", span_id="s1", parent_span_id=None,
            operation_name="op"
        )
        time.sleep(0.01)
        assert span.duration_ms > 0

    def test_span_set_attribute(self):
        span = Span(trace_id="t1", span_id="s1", parent_span_id=None, operation_name="op")
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_span_add_event(self):
        span = Span(trace_id="t1", span_id="s1", parent_span_id=None, operation_name="op")
        span.add_event("checkpoint", {"step": 1})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"

    def test_span_end(self):
        span = Span(trace_id="t1", span_id="s1", parent_span_id=None, operation_name="op")
        span.end("error")
        assert span.end_time is not None
        assert span.status == "error"

    def test_span_to_dict(self):
        span = Span(trace_id="t1", span_id="s1", parent_span_id="p1", operation_name="op")
        span.set_attribute("test", True)
        span.end()
        result = span.to_dict()
        assert result["trace_id"] == "t1"
        assert result["span_id"] == "s1"
        assert result["parent_span_id"] == "p1"
        assert result["attributes"]["test"] is True


class TestTracer:
    """Tests for Tracer."""

    @pytest.fixture
    def tracer(self):
        return Tracer(service_name="test-service")

    def test_start_span(self, tracer):
        span = tracer.start_span("test_operation")
        assert span.operation_name == "test_operation"
        assert span.attributes["service.name"] == "test-service"

    def test_start_span_with_parent(self, tracer):
        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent_span_id=parent.span_id)
        assert child.parent_span_id == parent.span_id

    def test_end_span(self, tracer):
        span = tracer.start_span("test")
        tracer.end_span(span)
        assert span.end_time is not None
        assert span in tracer._completed_spans

    def test_get_completed_spans(self, tracer):
        span1 = tracer.start_span("op1")
        span2 = tracer.start_span("op2")
        tracer.end_span(span1)
        tracer.end_span(span2)
        spans = tracer.get_completed_spans()
        assert len(spans) == 2

    def test_get_completed_spans_clear(self, tracer):
        span = tracer.start_span("op")
        tracer.end_span(span)
        tracer.get_completed_spans(clear=True)
        assert len(tracer._completed_spans) == 0


class TestTracedDecorator:
    """Tests for @traced decorator."""

    @pytest.mark.asyncio
    async def test_traced_async_function(self):
        @traced("test_async_op")
        async def async_function():
            await asyncio.sleep(0.01)
            return "result"
        result = await async_function()
        assert result == "result"
        tracer = get_tracer()
        spans = tracer.get_completed_spans(clear=True)
        assert any(s.operation_name == "test_async_op" for s in spans)

    def test_traced_sync_function(self):
        @traced("test_sync_op")
        def sync_function():
            return "sync_result"
        result = sync_function()
        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_traced_exception(self):
        @traced("failing_op")
        async def failing_function():
            raise ValueError("Test error")
        with pytest.raises(ValueError):
            await failing_function()
        tracer = get_tracer()
        spans = tracer.get_completed_spans(clear=True)
        failing_span = next(s for s in spans if s.operation_name == "failing_op")
        assert failing_span.status == "error"


class TestTimedDecorator:
    """Tests for @timed decorator."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        get_metrics_registry().reset()
        yield
        get_metrics_registry().reset()

    @pytest.mark.asyncio
    async def test_timed_async_function(self):
        @timed("async_duration")
        async def async_function():
            await asyncio.sleep(0.01)
            return "done"
        result = await async_function()
        assert result == "done"
        metrics = get_metrics_registry().get_all()
        assert "solace_async_duration" in metrics["histograms"]

    def test_timed_sync_function(self):
        @timed("sync_duration")
        def sync_function():
            time.sleep(0.01)
            return "done"
        result = sync_function()
        assert result == "done"
        metrics = get_metrics_registry().get_all()
        assert "solace_sync_duration" in metrics["histograms"]


class TestCountedDecorator:
    """Tests for @counted decorator."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        get_metrics_registry().reset()
        yield
        get_metrics_registry().reset()

    @pytest.mark.asyncio
    async def test_counted_async_function(self):
        @counted("async_calls")
        async def async_function():
            return "called"
        await async_function()
        await async_function()
        await async_function()
        metrics = get_metrics_registry().get_all()
        assert metrics["counters"]["solace_async_calls"] == 3

    def test_counted_sync_function(self):
        @counted("sync_calls")
        def sync_function():
            return "called"
        sync_function()
        sync_function()
        metrics = get_metrics_registry().get_all()
        assert metrics["counters"]["solace_sync_calls"] == 2


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_default(self):
        configure_logging()

    def test_configure_logging_custom(self):
        settings = ObservabilitySettings(
            log_level=LogLevel.DEBUG,
            log_format="console"
        )
        configure_logging(settings)


class TestGlobalSingletons:
    """Tests for global singleton getters."""

    def test_get_metrics_registry_singleton(self):
        reg1 = get_metrics_registry()
        reg2 = get_metrics_registry()
        assert reg1 is reg2

    def test_get_tracer_singleton(self):
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2

    def test_add_log_context(self):
        logger = add_log_context(user_id="123", request_id="456")
        assert logger is not None
