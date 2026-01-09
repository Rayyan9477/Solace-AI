"""
Solace-AI Safety Service Telemetry - OpenTelemetry instrumentation for observability.
Provides distributed tracing and metrics for ML components.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, TypeVar, ParamSpec

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import Status, StatusCode, SpanKind
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class TelemetryConfig(BaseSettings):
    """Configuration for OpenTelemetry telemetry."""

    enabled: bool = Field(default=True, description="Enable telemetry")
    service_name: str = Field(default="safety-service", description="Service name for tracing")
    service_version: str = Field(default="1.0.0", description="Service version")
    otlp_endpoint: str = Field(default="http://localhost:4317", description="OTLP gRPC endpoint")
    export_timeout_ms: int = Field(default=30000, ge=1000, le=60000, description="Export timeout")
    max_queue_size: int = Field(default=2048, ge=512, le=8192, description="Max span queue size")
    schedule_delay_ms: int = Field(default=5000, ge=1000, le=30000, description="Batch schedule delay")

    model_config = SettingsConfigDict(
        env_prefix="TELEMETRY_",
        env_file=".env",
        extra="ignore"
    )


class SafetyServiceTelemetry:
    """
    OpenTelemetry instrumentation for Safety Service.
    Provides tracing for ML components with graceful fallback when disabled.
    """

    _instance: "SafetyServiceTelemetry | None" = None
    _initialized: bool = False

    def __new__(cls, config: TelemetryConfig | None = None) -> "SafetyServiceTelemetry":
        """Singleton pattern for telemetry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        """Initialize telemetry with configuration."""
        if self._initialized:
            return

        self._config = config or TelemetryConfig()
        self._tracer: trace.Tracer | None = None
        self._provider: TracerProvider | None = None

        if self._config.enabled and OPENTELEMETRY_AVAILABLE:
            self._initialize_tracer()
        else:
            reason = "disabled" if not self._config.enabled else "opentelemetry_not_available"
            logger.info("telemetry_skipped", reason=reason)

        self._initialized = True

    def _initialize_tracer(self) -> None:
        """Initialize OpenTelemetry tracer with OTLP exporter."""
        try:
            resource = Resource.create({
                SERVICE_NAME: self._config.service_name,
                SERVICE_VERSION: self._config.service_version,
            })

            self._provider = TracerProvider(resource=resource)

            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=self._config.otlp_endpoint,
                insecure=True,
            )

            # Configure batch processor
            span_processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=self._config.max_queue_size,
                schedule_delay_millis=self._config.schedule_delay_ms,
                export_timeout_millis=self._config.export_timeout_ms,
            )

            self._provider.add_span_processor(span_processor)
            trace.set_tracer_provider(self._provider)

            self._tracer = trace.get_tracer(
                self._config.service_name,
                self._config.service_version
            )

            logger.info(
                "telemetry_initialized",
                service=self._config.service_name,
                endpoint=self._config.otlp_endpoint
            )

        except Exception as e:
            logger.error("telemetry_init_failed", error=str(e))
            self._tracer = None

    @property
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled and initialized."""
        return self._tracer is not None

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.INTERNAL
    ):
        """
        Create a traced span context manager.

        Args:
            name: Span name
            attributes: Optional span attributes
            kind: Span kind (INTERNAL, SERVER, CLIENT, etc.)

        Yields:
            The span object (or None if telemetry disabled)
        """
        if not self.is_enabled:
            yield None
            return

        with self._tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value) if not isinstance(value, (str, int, float, bool)) else value)
            yield span

    def record_error(self, span: Any, error: Exception, message: str | None = None) -> None:
        """Record an error on a span."""
        if span is None or not self.is_enabled:
            return

        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, message or str(error)))

    def add_event(self, span: Any, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to a span."""
        if span is None or not self.is_enabled:
            return

        span.add_event(name, attributes=attributes or {})

    def shutdown(self) -> None:
        """Shutdown telemetry and flush pending spans."""
        if self._provider:
            try:
                self._provider.shutdown()
                logger.info("telemetry_shutdown")
            except Exception as e:
                logger.error("telemetry_shutdown_failed", error=str(e))


# Global telemetry instance
_telemetry: SafetyServiceTelemetry | None = None


def get_telemetry(config: TelemetryConfig | None = None) -> SafetyServiceTelemetry:
    """Get or create the global telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = SafetyServiceTelemetry(config)
    return _telemetry


def traced(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
    record_args: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to trace a function with OpenTelemetry.

    Args:
        name: Span name (defaults to function name)
        attributes: Static span attributes
        record_args: Whether to record function arguments as attributes

    Returns:
        Decorated function with tracing
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            telemetry = get_telemetry()

            span_attrs = dict(attributes or {})
            span_attrs["function"] = func.__qualname__

            if record_args and args:
                span_attrs["args_count"] = len(args)
            if record_args and kwargs:
                span_attrs["kwargs_keys"] = ",".join(kwargs.keys())

            with telemetry.span(span_name, attributes=span_attrs) as span:
                try:
                    result = func(*args, **kwargs)
                    if span:
                        span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    telemetry.record_error(span, e)
                    raise

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            telemetry = get_telemetry()

            span_attrs = dict(attributes or {})
            span_attrs["function"] = func.__qualname__

            if record_args and args:
                span_attrs["args_count"] = len(args)
            if record_args and kwargs:
                span_attrs["kwargs_keys"] = ",".join(kwargs.keys())

            with telemetry.span(span_name, attributes=span_attrs) as span:
                try:
                    result = await func(*args, **kwargs)
                    if span:
                        span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    telemetry.record_error(span, e)
                    raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
