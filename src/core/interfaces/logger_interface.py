"""
Abstract interface for Logging providers.

This interface provides a contract for logging systems,
enabling flexible logging backends and structured logging.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class LogLevel(Enum):
    """Enum for log levels."""
    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogEntry:
    """Represents a log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    context: Optional[Dict[str, Any]] = None
    exception: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class LoggerConfig:
    """Configuration for logger providers."""
    level: LogLevel = LogLevel.INFO
    format_string: Optional[str] = None
    include_timestamp: bool = True
    include_correlation_id: bool = True
    structured_logging: bool = False
    additional_params: Optional[Dict[str, Any]] = None


class LoggerInterface(ABC):
    """
    Abstract base class for all logging providers.
    
    This interface ensures all logging providers implement
    consistent methods for logging and configuration.
    """
    
    def __init__(self, name: str, config: LoggerConfig):
        """Initialize the logger with configuration."""
        self.name = name
        self.config = config
        self.correlation_id = None
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the logging provider.
        
        Returns:
            bool: True if initialization was successful
        """
        pass
    
    @abstractmethod
    def log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """
        Log a message at the specified level.
        
        Args:
            level: Log level
            message: Log message
            context: Additional context data
            exception: Exception object if logging an error
        """
        pass
    
    @abstractmethod
    async def flush(self) -> None:
        """Flush any buffered log entries."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the logger and cleanup resources."""
        pass
    
    def trace(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a trace message."""
        self.log(LogLevel.TRACE, message, context)
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        self.log(LogLevel.DEBUG, message, context)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message."""
        self.log(LogLevel.INFO, message, context)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self.log(LogLevel.WARNING, message, context)
    
    def error(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """Log an error message."""
        self.log(LogLevel.ERROR, message, context, exception)
    
    def critical(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """Log a critical message."""
        self.log(LogLevel.CRITICAL, message, context, exception)
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set the correlation ID for tracing requests."""
        self.correlation_id = correlation_id
    
    def clear_correlation_id(self) -> None:
        """Clear the correlation ID."""
        self.correlation_id = None
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set persistent context data for all log entries."""
        if not hasattr(self, '_persistent_context'):
            self._persistent_context = {}
        self._persistent_context.update(context)
    
    def clear_context(self) -> None:
        """Clear persistent context data."""
        self._persistent_context = {}
    
    def _merge_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge persistent context with provided context."""
        merged = getattr(self, '_persistent_context', {}).copy()
        if context:
            merged.update(context)
        
        if self.correlation_id:
            merged['correlation_id'] = self.correlation_id
        
        return merged
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if a message should be logged based on current level."""
        return level.value >= self.config.level.value
    
    @property
    def is_initialized(self) -> bool:
        """Check if the logger is initialized."""
        return self._initialized
    
    @property
    def logger_name(self) -> str:
        """Get the logger name."""
        return self.name
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the logging provider.
        
        Returns:
            Dict containing health status
        """
        try:
            return {
                "status": "healthy",
                "logger_name": self.name,
                "initialized": self.is_initialized,
                "level": self.config.level.name,
                "structured_logging": self.config.structured_logging
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "logger_name": self.name,
                "error": str(e),
                "initialized": self.is_initialized
            }


class StructuredLoggerMixin:
    """
    Mixin to add structured logging capabilities.
    
    This mixin can be used with LoggerInterface implementations
    to provide structured logging functionality.
    """
    
    def log_structured(
        self,
        event: str,
        level: LogLevel = LogLevel.INFO,
        **kwargs
    ) -> None:
        """
        Log a structured event.
        
        Args:
            event: Event name
            level: Log level
            **kwargs: Event data
        """
        context = {
            'event': event,
            'event_data': kwargs,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.log(level, f"Event: {event}", context)
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
            **kwargs: Additional metrics
        """
        context = {
            'performance': {
                'operation': operation,
                'duration_ms': duration_ms,
                'success': success,
                **kwargs
            }
        }
        level = LogLevel.INFO if success else LogLevel.WARNING
        self.log(level, f"Performance: {operation} ({duration_ms}ms)", context)
    
    def log_security(
        self,
        event: str,
        severity: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log security events.
        
        Args:
            event: Security event name
            severity: Event severity (low, medium, high, critical)
            user_id: User ID if applicable
            **kwargs: Additional security context
        """
        context = {
            'security': {
                'event': event,
                'severity': severity,
                'user_id': user_id,
                **kwargs
            }
        }
        level = LogLevel.WARNING if severity in ['medium', 'high'] else LogLevel.CRITICAL
        self.log(level, f"Security: {event} [{severity}]", context)