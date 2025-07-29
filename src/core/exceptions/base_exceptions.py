"""
Base exception classes for the Contextual-Chatbot application.
"""

from typing import Dict, Any, Optional


class ChatbotBaseException(Exception):
    """
    Base exception class for all chatbot-related exceptions.
    
    Provides common functionality for error handling, including
    error codes, context information, and structured error data.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize the base exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            context: Additional context information
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary representation.
        
        Returns:
            Dictionary containing exception details
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'cause': str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """String representation of the exception."""
        base_msg = f"{self.error_code}: {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" ({context_str})"
        if self.cause:
            base_msg += f" [Caused by: {self.cause}]"
        return base_msg


class ConfigurationError(ChatbotBaseException):
    """
    Exception raised for configuration-related errors.
    
    This includes missing configuration values, invalid configuration,
    or configuration validation failures.
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = str(config_value)
        
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class InitializationError(ChatbotBaseException):
    """
    Exception raised when component initialization fails.
    
    This includes failures in setting up dependencies, connecting to
    external services, or validating component state.
    """
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if component:
            context['component'] = component
        
        super().__init__(
            message,
            error_code="INIT_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class ValidationError(ChatbotBaseException):
    """
    Exception raised for validation failures.
    
    This includes input validation, data validation, and business rule
    validation failures.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        validation_rules: Optional[list] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if field:
            context['field'] = field
        if value is not None:
            context['value'] = str(value)
        if validation_rules:
            context['validation_rules'] = validation_rules
        
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )