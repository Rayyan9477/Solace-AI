"""
OpenAI LLM Provider implementation.

This module provides a concrete implementation of the LLMInterface
for OpenAI's GPT models.
"""

import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import json

from ...core.interfaces.llm_interface import (
    LLMInterface, 
    LLMConfig, 
    Message, 
    MessageRole, 
    LLMResponse
)
from ...core.exceptions.llm_exceptions import (
    LLMProviderError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMInvalidRequestError
)

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class OpenAIProvider(LLMInterface):
    """
    OpenAI provider implementation.
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the OpenAI provider."""
        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                "OpenAI library not available. Install with: pip install openai",
                provider="openai"
            )
        
        super().__init__(config)
        self.client = None
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI provider."""
        try:
            # Create async OpenAI client
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=60.0
            )
            
            # Test connection
            test_response = await self._test_connection()
            if not test_response:
                return False
            
            self._initialized = True
            return True
            
        except Exception as e:
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                raise LLMAuthenticationError(
                    f"Invalid OpenAI API key: {str(e)}",
                    provider="openai",
                    auth_method="api_key"
                )
            else:
                raise LLMProviderError(
                    f"Failed to initialize OpenAI provider: {str(e)}",
                    provider="openai",
                    model=self.config.model_name
                )
    
    async def _test_connection(self) -> bool:
        """Test connection to OpenAI API."""
        try:
            # Try a minimal chat completion
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            return response.choices[0].message.content is not None
        except (LLMConnectionError, LLMAuthenticationError, LLMRateLimitError,
                OSError, TimeoutError, RuntimeError):
            return False
    
    def _convert_messages_to_openai_format(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert messages to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            role_map = {
                MessageRole.USER: "user",
                MessageRole.ASSISTANT: "assistant",
                MessageRole.SYSTEM: "system"
            }
            
            openai_messages.append({
                "role": role_map.get(message.role, "user"),
                "content": message.content
            })
        
        return openai_messages
    
    async def generate_response(
        self, 
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenAI."""
        if not self._initialized:
            raise LLMProviderError("Provider not initialized", provider="openai")
        
        try:
            openai_messages = self._convert_messages_to_openai_format(messages)
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                **self.config.additional_params or {}
            )
            
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Extract usage information
            usage = {}
            if response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            return LLMResponse(
                content=content,
                metadata={
                    'model': response.model,
                    'provider': 'openai',
                    'generation_config': {
                        'temperature': self.config.temperature,
                        'max_tokens': self.config.max_tokens,
                        'top_p': self.config.top_p
                    }
                },
                usage=usage,
                finish_reason=choice.finish_reason
            )
            
        except Exception as e:
            error_str = str(e).lower()
            
            if 'rate limit' in error_str or 'quota' in error_str:
                raise LLMRateLimitError(
                    f"OpenAI rate limit exceeded: {str(e)}",
                    provider="openai"
                )
            elif 'invalid' in error_str or 'bad request' in error_str:
                raise LLMInvalidRequestError(
                    f"Invalid request to OpenAI: {str(e)}",
                    provider="openai"
                )
            elif 'connection' in error_str or 'network' in error_str:
                raise LLMConnectionError(
                    f"Connection error with OpenAI: {str(e)}",
                    provider="openai"
                )
            else:
                raise LLMProviderError(
                    f"OpenAI generation error: {str(e)}",
                    provider="openai",
                    model=self.config.model_name
                )
    
    async def generate_streaming_response(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response using OpenAI."""
        if not self._initialized:
            raise LLMProviderError("Provider not initialized", provider="openai")
        
        try:
            openai_messages = self._convert_messages_to_openai_format(messages)
            
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,
                **self.config.additional_params or {}
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise LLMProviderError(
                f"OpenAI streaming error: {str(e)}",
                provider="openai",
                model=self.config.model_name
            )
    
    async def validate_connection(self) -> bool:
        """Validate connection to OpenAI API."""
        if not self._initialized:
            return False
        
        try:
            return await self._test_connection()
        except (LLMConnectionError, LLMAuthenticationError, LLMRateLimitError,
                OSError, TimeoutError, RuntimeError):
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model."""
        return {
            'provider': 'openai',
            'model_name': self.config.model_name,
            'max_tokens': self.config.max_tokens,
            'supports_streaming': True,
            'supports_conversation': True,
            'supports_system_messages': True,
            'temperature_range': [0.0, 2.0],
            'top_p_range': [0.0, 1.0],
            'context_window': self._get_context_window()
        }
    
    def _get_context_window(self) -> int:
        """Get context window size for the model."""
        context_windows = {
            'gpt-4': 8192,
            'gpt-4-32k': 32768,
            'gpt-4-turbo': 128000,
            'gpt-4o': 128000,
            'gpt-3.5-turbo': 4096,
            'gpt-3.5-turbo-16k': 16384
        }
        
        return context_windows.get(self.config.model_name, 4096)
    
    async def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation for OpenAI models (approximately 4 characters per token)
        return len(text) // 4
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "OpenAIProvider"