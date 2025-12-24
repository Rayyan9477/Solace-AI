"""
Comprehensive Error Handling and Logging System
Implements structured error handling, logging, and monitoring for Solace-AI
"""

import sys
import json
import logging
import traceback
import contextvars
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Callable, Union, List, Type
from functools import wraps
from enum import Enum
import time
from pathlib import Path

# Context variables for request tracking
request_id = contextvars.ContextVar('request_id', default=None)
user_id = contextvars.ContextVar('user_id', default=None)
session_id = contextvars.ContextVar('session_id', default=None)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    SYSTEM = "system"
    SECURITY = "security"
    HIPAA_COMPLIANCE = "hipaa_compliance"

class ErrorType(Enum):
    """Legacy error types - kept for compatibility"""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    MODEL_ERROR = "model_error"
    DATA_ERROR = "data_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorDetails:
    """Detailed error information"""
    error_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime
    component: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    resolution_status: str = "unresolved"
    resolution_notes: str = ""

class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, field: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.field = field
        self.context = context or {}

class ProcessingError(Exception):
    """Custom processing error"""
    def __init__(self, message: str, component: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.component = component
        self.context = context or {}

class ErrorHandler:
    """Central error handling and logging system"""

    # Maximum number of errors to keep in history to prevent memory leak
    MAX_ERROR_HISTORY_SIZE = 1000

    def __init__(self):
        self.error_history: List[ErrorDetails] = []
        self.error_counts: Dict[str, int] = {}
        self.suppressed_errors: set = set()
        
    def handle_error(self, 
                    error: Exception, 
                    component: str = "unknown",
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    context: Dict[str, Any] = None) -> ErrorDetails:
        """Handle and log an error"""
        
        # Determine error type
        error_type = self._classify_error(error)
        
        # Create error details
        error_details = ErrorDetails(
            error_id=self._generate_error_id(),
            error_type=error_type,
            severity=severity,
            message=str(error),
            timestamp=datetime.utcnow(),
            component=component,
            stack_trace=traceback.format_exc(),
            context=context or {},
            user_id=user_id,
            session_id=session_id
        )
        
        # Log error
        self._log_error(error_details)

        # Store error with bounded history to prevent memory leak (SEC-MEMORY-001)
        self.error_history.append(error_details)
        # Trim old errors if history exceeds max size
        if len(self.error_history) > self.MAX_ERROR_HISTORY_SIZE:
            # Remove oldest 10% of errors when limit is reached
            trim_count = self.MAX_ERROR_HISTORY_SIZE // 10
            self.error_history = self.error_history[trim_count:]

        # Update error counts
        error_key = f"{component}:{error_type.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Trigger alerts for critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._trigger_alert(error_details)
        
        return error_details
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error based on type and message"""
        if isinstance(error, ValidationError):
            return ErrorType.VALIDATION_ERROR
        elif isinstance(error, ProcessingError):
            return ErrorType.PROCESSING_ERROR
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorType.NETWORK_ERROR
        elif isinstance(error, PermissionError):
            return ErrorType.PERMISSION_ERROR
        elif isinstance(error, (MemoryError, ResourceWarning)):
            return ErrorType.RESOURCE_ERROR
        elif "model" in str(error).lower():
            return ErrorType.MODEL_ERROR
        elif "data" in str(error).lower():
            return ErrorType.DATA_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        import uuid
        return f"ERR_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level"""
        log_message = f"[{error_details.error_id}] {error_details.component}: {error_details.message}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _trigger_alert(self, error_details: ErrorDetails):
        """Trigger alert for critical errors"""
        # This would integrate with alerting systems (email, Slack, PagerDuty, etc.)
        logger.critical(f"CRITICAL ERROR ALERT: {error_details.error_id} - {error_details.message}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {"total_errors": 0}
        
        recent_errors = [e for e in self.error_history if 
                        (datetime.utcnow() - e.timestamp).days <= 7]
        
        error_by_type = {}
        error_by_severity = {}
        error_by_component = {}
        
        for error in recent_errors:
            error_by_type[error.error_type.value] = error_by_type.get(error.error_type.value, 0) + 1
            error_by_severity[error.severity.name] = error_by_severity.get(error.severity.name, 0) + 1
            error_by_component[error.component] = error_by_component.get(error.component, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_by_type": error_by_type,
            "error_by_severity": error_by_severity,
            "error_by_component": error_by_component,
            "most_common_errors": sorted(self.error_counts.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
        }

# Global error handler instance
error_handler = ErrorHandler()

def handle_exceptions(component: str = None, 
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     reraise: bool = False,
                     default_return: Any = None):
    """Decorator for handling exceptions in functions"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_details = error_handler.handle_error(
                    error=e,
                    component=component or func.__name__,
                    severity=severity,
                    context={"args": str(args), "kwargs": str(kwargs)}
                )
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator

class InputValidator:
    """Comprehensive input validation system"""
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate that required fields are present"""
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        return missing_fields
    
    @staticmethod
    def validate_data_types(data: Dict[str, Any], type_mapping: Dict[str, Type]) -> List[str]:
        """Validate data types"""
        type_errors = []
        for field, expected_type in type_mapping.items():
            if field in data and data[field] is not None:
                if not isinstance(data[field], expected_type):
                    type_errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
        return type_errors
    
    @staticmethod
    def validate_string_constraints(data: Dict[str, Any], 
                                  constraints: Dict[str, Dict[str, Any]]) -> List[str]:
        """Validate string constraints (length, pattern, etc.)"""
        constraint_errors = []
        
        for field, field_constraints in constraints.items():
            if field not in data or not isinstance(data[field], str):
                continue
            
            value = data[field]
            
            # Check minimum length
            if 'min_length' in field_constraints and len(value) < field_constraints['min_length']:
                constraint_errors.append(f"Field '{field}' must be at least {field_constraints['min_length']} characters")
            
            # Check maximum length
            if 'max_length' in field_constraints and len(value) > field_constraints['max_length']:
                constraint_errors.append(f"Field '{field}' must be at most {field_constraints['max_length']} characters")
            
            # Check pattern
            if 'pattern' in field_constraints:
                import re
                if not re.match(field_constraints['pattern'], value):
                    constraint_errors.append(f"Field '{field}' does not match required pattern")
            
            # Check allowed values
            if 'allowed_values' in field_constraints and value not in field_constraints['allowed_values']:
                constraint_errors.append(f"Field '{field}' must be one of: {field_constraints['allowed_values']}")
        
        return constraint_errors
    
    @staticmethod
    def validate_numeric_constraints(data: Dict[str, Any], 
                                   constraints: Dict[str, Dict[str, Any]]) -> List[str]:
        """Validate numeric constraints"""
        constraint_errors = []
        
        for field, field_constraints in constraints.items():
            if field not in data or not isinstance(data[field], (int, float)):
                continue
            
            value = data[field]
            
            # Check minimum value
            if 'min_value' in field_constraints and value < field_constraints['min_value']:
                constraint_errors.append(f"Field '{field}' must be at least {field_constraints['min_value']}")
            
            # Check maximum value
            if 'max_value' in field_constraints and value > field_constraints['max_value']:
                constraint_errors.append(f"Field '{field}' must be at most {field_constraints['max_value']}")
            
            # Check if integer required
            if field_constraints.get('integer_only', False) and not isinstance(value, int):
                constraint_errors.append(f"Field '{field}' must be an integer")
        
        return constraint_errors

def validate_input(schema: Dict[str, Any]):
    """Decorator for input validation"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract data to validate (assumes first argument or 'data' keyword)
            data = None
            if args:
                if isinstance(args[0], dict):
                    data = args[0]
                elif hasattr(args[0], '__dict__'):
                    data = args[0].__dict__
            
            if 'data' in kwargs:
                data = kwargs['data']
            
            if data is None:
                raise ValidationError("No data provided for validation")
            
            errors = []
            
            # Validate required fields
            if 'required' in schema:
                errors.extend(InputValidator.validate_required_fields(data, schema['required']))
            
            # Validate data types
            if 'types' in schema:
                errors.extend(InputValidator.validate_data_types(data, schema['types']))
            
            # Validate string constraints
            if 'string_constraints' in schema:
                errors.extend(InputValidator.validate_string_constraints(data, schema['string_constraints']))
            
            # Validate numeric constraints
            if 'numeric_constraints' in schema:
                errors.extend(InputValidator.validate_numeric_constraints(data, schema['numeric_constraints']))
            
            if errors:
                raise ValidationError(f"Validation failed: {'; '.join(errors)}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

class RetryHandler:
    """Retry mechanism for transient failures"""
    
    @staticmethod
    def retry(max_attempts: int = 3, 
             delay: float = 1.0, 
             backoff_factor: float = 2.0,
             exceptions: tuple = (Exception,)):
        """Retry decorator"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                attempt = 0
                current_delay = delay
                
                while attempt < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        attempt += 1
                        if attempt >= max_attempts:
                            error_handler.handle_error(
                                error=e,
                                component=func.__name__,
                                severity=ErrorSeverity.HIGH,
                                context={"max_attempts_reached": True, "attempts": attempt}
                            )
                            raise
                        
                        logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                
                return None
            
            return wrapper
        return decorator

class CircuitBreaker:
    """Circuit breaker pattern for failing services"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise ProcessingError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e