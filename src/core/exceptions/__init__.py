"""
Core exceptions for the Contextual-Chatbot application.

This module defines all custom exceptions used throughout the application,
providing structured error handling and better debugging information.
"""

from .base_exceptions import (
    ChatbotBaseException,
    ConfigurationError,
    InitializationError,
    ValidationError
)

from .factory_exceptions import (
    ProviderNotFoundError,
    ProviderInitializationError,
    FactoryError
)

from .llm_exceptions import (
    LLMProviderError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMInvalidRequestError
)

from .storage_exceptions import (
    StorageError,
    StorageConnectionError,
    DocumentNotFoundError,
    StoragePermissionError
)

from .agent_exceptions import (
    AgentError,
    AgentInitializationError,
    AgentProcessingError,
    AgentDependencyError
)

from .security_exceptions import (
    SecurityException,
    AuthenticationError,
    AuthorizationError,
    InputValidationError,
    InjectionAttackDetected,
    XSSAttackDetected,
    SecretValidationError,
    SecretRotationRequired,
    RateLimitExceeded,
    EncryptionError,
    DataExposureRisk,
    CircuitBreakerOpen
)

__all__ = [
    # Base exceptions
    'ChatbotBaseException',
    'ConfigurationError',
    'InitializationError',
    'ValidationError',

    # Factory exceptions
    'ProviderNotFoundError',
    'ProviderInitializationError',
    'FactoryError',

    # LLM exceptions
    'LLMProviderError',
    'LLMConnectionError',
    'LLMAuthenticationError',
    'LLMRateLimitError',
    'LLMInvalidRequestError',

    # Storage exceptions
    'StorageError',
    'StorageConnectionError',
    'DocumentNotFoundError',
    'StoragePermissionError',

    # Agent exceptions
    'AgentError',
    'AgentInitializationError',
    'AgentProcessingError',
    'AgentDependencyError',

    # Security exceptions
    'SecurityException',
    'AuthenticationError',
    'AuthorizationError',
    'InputValidationError',
    'InjectionAttackDetected',
    'XSSAttackDetected',
    'SecretValidationError',
    'SecretRotationRequired',
    'RateLimitExceeded',
    'EncryptionError',
    'DataExposureRisk',
    'CircuitBreakerOpen'
]