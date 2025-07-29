"""
LLM provider-related exception classes.
"""

from typing import Any, Optional, Dict
from .base_exceptions import ChatbotBaseException


class LLMProviderError(ChatbotBaseException):
    """
    Base exception for LLM provider-related errors.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if provider:
            context['provider'] = provider
        if model:
            context['model'] = model
        
        super().__init__(
            message,
            error_code="LLM_PROVIDER_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class LLMConnectionError(LLMProviderError):
    """
    Exception raised when connection to LLM provider fails.
    """
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if endpoint:
            context['endpoint'] = endpoint
        
        super().__init__(
            message,
            error_code="LLM_CONNECTION_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class LLMAuthenticationError(LLMProviderError):
    """
    Exception raised for authentication failures with LLM providers.
    """
    
    def __init__(
        self,
        message: str,
        auth_method: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if auth_method:
            context['auth_method'] = auth_method
        
        super().__init__(
            message,
            error_code="LLM_AUTH_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class LLMRateLimitError(LLMProviderError):
    """
    Exception raised when LLM provider rate limits are exceeded.
    """
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        current_usage: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if retry_after:
            context['retry_after_seconds'] = retry_after
        if current_usage:
            context['current_usage'] = current_usage
        
        super().__init__(
            message,
            error_code="LLM_RATE_LIMIT_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class LLMInvalidRequestError(LLMProviderError):
    """
    Exception raised for invalid requests to LLM providers.
    """
    
    def __init__(
        self,
        message: str,
        request_data: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[list] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if request_data:
            # Don't include full request data for security, just keys
            context['request_keys'] = list(request_data.keys())
        if validation_errors:
            context['validation_errors'] = validation_errors
        
        super().__init__(
            message,
            error_code="LLM_INVALID_REQUEST",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )