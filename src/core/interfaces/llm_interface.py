"""
Abstract interface for Language Model providers.

This interface ensures all LLM providers implement the same methods,
allowing easy swapping between different providers (Gemini, OpenAI, Anthropic, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    """Enum for message roles in conversations."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """Represents a message in a conversation."""
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    metadata: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    top_k: Optional[int] = None
    additional_params: Optional[Dict[str, Any]] = None


class LLMInterface(ABC):
    """
    Abstract base class for all Language Model providers.
    
    This interface defines the contract that all LLM providers must implement,
    ensuring consistent behavior regardless of the underlying provider.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM provider with configuration."""
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the LLM provider.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse: The generated response
            
        Raises:
            LLMProviderError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters
            
        Yields:
            str: Chunks of the generated response
            
        Raises:
            LLMProviderError: If generation fails
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate that the connection to the LLM provider is working.
        
        Returns:
            bool: True if connection is valid, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict containing model information like name, version, capabilities, etc.
        """
        pass
    
    @abstractmethod
    async def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            int: Estimated number of tokens
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._initialized
    
    @property
    def provider_name(self) -> str:
        """Get the name of the provider."""
        return self.__class__.__name__
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.
        
        Returns:
            Dict containing health status information
        """
        try:
            is_valid = await self.validate_connection()
            model_info = self.get_model_info()
            
            return {
                "status": "healthy" if is_valid else "unhealthy",
                "provider": self.provider_name,
                "initialized": self.is_initialized,
                "model_info": model_info,
                "config": {
                    "model_name": self.config.model_name,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "error": str(e),
                "initialized": self.is_initialized
            }