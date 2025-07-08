"""
LLM Module

Provides a unified interface to various Large Language Models (LLMs)
and handles model loading, caching, and inference.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import os
import time
import json
from pathlib import Path

from src.components.base_module import Module
from src.config.settings import AppConfig

class LLMModule(Module):
    """
    LLM Module for the Contextual-Chatbot.
    
    Provides a unified interface to various LLM backends:
    - Gemini API
    - Local LLMs
    - Others as needed
    
    Handles:
    - Model loading and initialization
    - Caching of results
    - Fallback strategies
    - Token usage tracking
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize the module"""
        super().__init__(module_id, config)
        self.llm = None
        self.api_key = None
        self.cache_dir = None
        self.model_name = None
        self.max_tokens = 4096
        self.supports_streaming = False
        self.supports_chat = True
        self.has_vision_capabilities = False
        
        # Initialize config
        self._load_config()
    
    def _load_config(self):
        """Load configuration values"""
        if not self.config:
            return
            
        self.model_name = self.config.get("model_name", "gemini-pro")
        self.api_key = self.config.get("api_key", os.environ.get("GEMINI_API_KEY"))
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.cache_dir = self.config.get("cache_dir", os.path.join(
            Path(__file__).parents[2], "models", "cache"
        ))
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize the module"""
        await super().initialize()
        
        try:
            # Check if model is Gemini
            if self.model_name.startswith("gemini"):
                success = await self._initialize_gemini()
            else:
                # Default to Gemini if unspecified
                self.model_name = "gemini-pro"
                success = await self._initialize_gemini()
            
            if success:
                self._register_services()
                return True
            else:
                self.health_status = "degraded"
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM module: {str(e)}")
            self.health_status = "failed"
            return False
    
    async def _initialize_gemini(self) -> bool:
        """Initialize Gemini API"""
        try:
            # Attempt to import Gemini-specific modules
            try:
                from src.models.gemini_llm import GeminiLLM
                
                # Initialize Gemini
                self.llm = GeminiLLM(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    cache_dir=self.cache_dir
                )
                
                # Update capabilities based on model
                if self.model_name == "gemini-pro-vision":
                    self.has_vision_capabilities = True
                
                self.supports_streaming = True
                self.supports_chat = True
                
                self.logger.info(f"Initialized Gemini LLM with model {self.model_name}")
                return True
                
            except ImportError:
                self.logger.error("Gemini modules not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {str(e)}")
            return False
    
    def _register_services(self):
        """Register services provided by this module"""
        self.expose_service("generate_text", self.generate_text)
        self.expose_service("generate_chat_response", self.generate_chat_response)
        self.expose_service("get_embedding", self.get_embedding)
        
        if self.has_vision_capabilities:
            self.expose_service("generate_vision_response", self.generate_vision_response)
    
    async def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: The text prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not self.initialized:
            self.logger.warning("LLM module not initialized")
            return "I'm sorry, but the language model is not available right now."
        
        try:
            if hasattr(self.llm, "generate_text"):
                response = await self.llm.generate_text(prompt, max_tokens or self.max_tokens)
                return response
            else:
                self.logger.error("LLM does not support text generation")
                return "I'm sorry, but the language model does not support this operation."
        except Exception as e:
            self.logger.error(f"Error in text generation: {str(e)}")
            return f"I'm sorry, but there was an error processing your request."
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], 
                                   max_tokens: Optional[int] = None) -> str:
        """
        Generate a response from a chat history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        if not self.initialized:
            self.logger.warning("LLM module not initialized")
            return "I'm sorry, but the language model is not available right now."
            
        if not self.supports_chat:
            # Fallback to text generation with formatted prompt
            return await self._fallback_chat_to_text(messages, max_tokens)
        
        try:
            if hasattr(self.llm, "generate_chat_response"):
                response = await self.llm.generate_chat_response(messages, max_tokens or self.max_tokens)
                return response
            else:
                return await self._fallback_chat_to_text(messages, max_tokens)
        except Exception as e:
            self.logger.error(f"Error in chat response generation: {str(e)}")
            return f"I'm sorry, but there was an error processing your request."
    
    async def _fallback_chat_to_text(self, messages: List[Dict[str, str]], 
                                   max_tokens: Optional[int] = None) -> str:
        """Fallback chat to text generation"""
        prompt = self._format_chat_as_text(messages)
        return await self.generate_text(prompt, max_tokens)
    
    def _format_chat_as_text(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages as a text prompt"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted) + "\nAssistant: "
    
    async def generate_vision_response(self, image_path: str, prompt: str,
                                     max_tokens: Optional[int] = None) -> str:
        """
        Generate a response based on an image and prompt
        
        Args:
            image_path: Path to image file
            prompt: Text prompt describing the query about the image
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        if not self.initialized:
            self.logger.warning("LLM module not initialized")
            return "I'm sorry, but the vision model is not available right now."
            
        if not self.has_vision_capabilities:
            self.logger.warning("Vision capabilities not available in this model")
            return "I'm sorry, but vision capabilities are not available in the current model."
        
        try:
            if hasattr(self.llm, "generate_vision_response"):
                response = await self.llm.generate_vision_response(
                    image_path, prompt, max_tokens or self.max_tokens
                )
                return response
            else:
                self.logger.error("LLM does not support vision")
                return "I'm sorry, but the model does not support vision capabilities."
        except Exception as e:
            self.logger.error(f"Error in vision response generation: {str(e)}")
            return f"I'm sorry, but there was an error processing the image."
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for text
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not self.initialized:
            self.logger.warning("LLM module not initialized")
            return []
        
        try:
            if hasattr(self.llm, "get_embedding"):
                embedding = await self.llm.get_embedding(text)
                return embedding
            else:
                self.logger.error("LLM does not support embeddings")
                return []
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            return []
    
    async def shutdown(self) -> bool:
        """Shutdown the module"""
        if hasattr(self.llm, "shutdown"):
            await self.llm.shutdown()
        
        return await super().shutdown()
