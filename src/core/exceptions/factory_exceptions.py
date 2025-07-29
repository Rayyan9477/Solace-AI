"""
Factory-related exception classes.
"""

from typing import Any, Optional, Dict
from .base_exceptions import ChatbotBaseException


class FactoryError(ChatbotBaseException):
    """
    Base exception for factory-related errors.
    """
    
    def __init__(
        self,
        message: str,
        factory_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if factory_type:
            context['factory_type'] = factory_type
        
        super().__init__(
            message,
            error_code="FACTORY_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class ProviderNotFoundError(FactoryError):
    """
    Exception raised when a requested provider is not registered or available.
    """
    
    def __init__(
        self,
        message: str,
        provider_type: Optional[str] = None,
        available_providers: Optional[list] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if provider_type:
            context['requested_provider'] = provider_type
        if available_providers:
            context['available_providers'] = available_providers
        
        super().__init__(
            message,
            error_code="PROVIDER_NOT_FOUND",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class ProviderInitializationError(FactoryError):
    """
    Exception raised when provider initialization fails.
    """
    
    def __init__(
        self,
        message: str,
        provider_type: Optional[str] = None,
        initialization_step: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if provider_type:
            context['provider_type'] = provider_type
        if initialization_step:
            context['initialization_step'] = initialization_step
        
        super().__init__(
            message,
            error_code="PROVIDER_INIT_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )