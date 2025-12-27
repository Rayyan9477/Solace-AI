"""
Google Gemini LLM Provider implementation.

This module provides a concrete implementation of the LLMInterface
for Google's Gemini models.
"""

import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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


class GeminiProvider(LLMInterface):
    """
    Google Gemini provider implementation.
    
    Supports Gemini 2.0 Flash and other Gemini models through the
    Google Generative AI API.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the Gemini provider."""
        super().__init__(config)
        self.client = None
        self.model = None
        self._token_count_cache = {}
    
    async def initialize(self) -> bool:
        """Initialize the Gemini provider."""
        try:
            # Configure the API
            genai.configure(api_key=self.config.api_key)
            
            # Create the model instance
            generation_config = genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k
            )
            
            # Configure safety settings (less restrictive for mental health context)
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Test connection
            test_response = await self._test_connection()
            if not test_response:
                return False
            
            self._initialized = True
            return True
            
        except Exception as e:
            if "API_KEY" in str(e).upper():
                raise LLMAuthenticationError(
                    f"Invalid Gemini API key: {str(e)}",
                    provider="gemini",
                    auth_method="api_key"
                )
            else:
                raise LLMProviderError(
                    f"Failed to initialize Gemini provider: {str(e)}",
                    provider="gemini",
                    model=self.config.model_name
                )
    
    async def _test_connection(self) -> bool:
        """Test connection to Gemini API."""
        try:
            # Try a simple generation request
            response = self.model.generate_content("Test connection")
            return hasattr(response, 'text') and response.text is not None
        except (LLMConnectionError, LLMAuthenticationError, LLMRateLimitError,
                OSError, TimeoutError, RuntimeError):
            return False
    
    def _convert_messages_to_gemini_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Gemini format."""
        gemini_messages = []
        
        for message in messages:
            role = "user" if message.role == MessageRole.USER else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": message.content}]
            })
        
        return gemini_messages
    
    async def generate_response(
        self, 
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate a response using Gemini."""
        if not self._initialized:
            raise LLMProviderError("Provider not initialized", provider="gemini")
        
        try:
            # Handle single message vs conversation
            if len(messages) == 1:
                # Single message - use generate_content
                response = self.model.generate_content(messages[0].content)
            else:
                # Multi-turn conversation - use chat
                chat = self.model.start_chat(history=[])
                
                # Add conversation history (skip the last message)
                for message in messages[:-1]:
                    if message.role == MessageRole.USER:
                        chat.send_message(message.content)
                
                # Send the latest message
                response = chat.send_message(messages[-1].content)
            
            # Extract response content
            if hasattr(response, 'text') and response.text:
                content = response.text
            else:
                raise LLMProviderError(
                    "No text content in Gemini response",
                    provider="gemini"
                )
            
            # Extract usage information if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    'prompt_tokens': response.usage_metadata.prompt_token_count,
                    'completion_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                }
            
            # Extract finish reason
            finish_reason = None
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
            
            return LLMResponse(
                content=content,
                metadata={
                    'model': self.config.model_name,
                    'provider': 'gemini',
                    'generation_config': {
                        'temperature': self.config.temperature,
                        'max_tokens': self.config.max_tokens,
                        'top_p': self.config.top_p,
                        'top_k': self.config.top_k
                    }
                },
                usage=usage,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            error_str = str(e).lower()
            
            if 'rate limit' in error_str or 'quota' in error_str:
                raise LLMRateLimitError(
                    f"Gemini rate limit exceeded: {str(e)}",
                    provider="gemini"
                )
            elif 'invalid' in error_str or 'bad request' in error_str:
                raise LLMInvalidRequestError(
                    f"Invalid request to Gemini: {str(e)}",
                    provider="gemini"
                )
            elif 'connection' in error_str or 'network' in error_str:
                raise LLMConnectionError(
                    f"Connection error with Gemini: {str(e)}",
                    provider="gemini"
                )
            else:
                raise LLMProviderError(
                    f"Gemini generation error: {str(e)}",
                    provider="gemini",
                    model=self.config.model_name
                )
    
    async def generate_streaming_response(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Gemini."""
        if not self._initialized:
            raise LLMProviderError("Provider not initialized", provider="gemini")
        
        try:
            # For streaming, we'll use the latest message
            prompt = messages[-1].content if messages else ""
            
            # Generate streaming response
            response = self.model.generate_content(
                prompt,
                stream=True
            )
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            raise LLMProviderError(
                f"Gemini streaming error: {str(e)}",
                provider="gemini",
                model=self.config.model_name
            )
    
    async def validate_connection(self) -> bool:
        """Validate connection to Gemini API."""
        if not self._initialized:
            return False
        
        try:
            return await self._test_connection()
        except (LLMConnectionError, LLMAuthenticationError, LLMRateLimitError,
                OSError, TimeoutError, RuntimeError):
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini model."""
        return {
            'provider': 'gemini',
            'model_name': self.config.model_name,
            'max_tokens': self.config.max_tokens,
            'supports_streaming': True,
            'supports_conversation': True,
            'supports_system_messages': False,  # Gemini doesn't have explicit system role
            'temperature_range': [0.0, 2.0],
            'top_p_range': [0.0, 1.0],
            'top_k_range': [1, 40]
        }
    
    async def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not self._initialized:
            return len(text.split()) * 1.3  # Rough estimate
        
        # Use cache to avoid repeated API calls
        if text in self._token_count_cache:
            return self._token_count_cache[text]
        
        try:
            # Use Gemini's count_tokens method if available
            if hasattr(self.model, 'count_tokens'):
                result = self.model.count_tokens(text)
                token_count = result.total_tokens
            else:
                # Fallback estimation (roughly 1.3 tokens per word for English)
                token_count = int(len(text.split()) * 1.3)
            
            # Cache the result
            self._token_count_cache[text] = token_count
            
            # Limit cache size
            if len(self._token_count_cache) > 100:
                # Remove oldest entries
                items = list(self._token_count_cache.items())
                self._token_count_cache = dict(items[-50:])
            
            return token_count

        except (AttributeError, TypeError, ValueError, RuntimeError):
            # Fallback to word-based estimation
            return int(len(text.split()) * 1.3)
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "GeminiProvider"