"""
DEPRECATED: This module has been moved to safety_service.src.infrastructure.telemetry.
This file exists only for backward compatibility. Import from infrastructure instead.
"""
import warnings
warnings.warn(
    "safety_service.src.observability is deprecated. Use safety_service.src.infrastructure.telemetry instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from safety_service.src.infrastructure.telemetry import (
    TelemetryConfig,
    Telemetry as SafetyServiceTelemetry,
    get_telemetry,
    traced,
)

__all__ = [
    "TelemetryConfig",
    "SafetyServiceTelemetry",
    "get_telemetry",
    "traced",
]
