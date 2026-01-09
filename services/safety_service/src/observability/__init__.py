"""
Solace-AI Safety Service Observability Module.
Provides OpenTelemetry instrumentation for tracing and metrics.
"""
from safety_service.src.observability.telemetry import (
    SafetyServiceTelemetry,
    TelemetryConfig,
    get_telemetry,
    traced,
)

__all__ = [
    "SafetyServiceTelemetry",
    "TelemetryConfig",
    "get_telemetry",
    "traced",
]
