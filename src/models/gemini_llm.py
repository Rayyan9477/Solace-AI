"""
Google Gemini 2.0 LLM integration for Contextual-Chatbot.

This module provides adapters for using Google's Gemini 2.0 API with the application.
"""

import os
import logging
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
from langchain.schema.language_model import BaseLanguageModel
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

from src.config.settings import AppConfig

logger = logging.getLogger(__name__)
ERROR_TEXT = "I'm sorry, I encountered an error processing your request."

class GeminiLLM(LLM):
    """
    LangChain adapter for Google Gemini 2.0 API
    
    This class implements the LangChain LLM interface for Google Gemini 2.0 API.
    """
    
    model_name: str = ""
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 64
    safety_settings: Optional[List[Dict[str, Any]]] = None
    
    def __init__(self, **kwargs):
        """Initialize the Gemini LLM"""
        super().__init__(**kwargs)
        
        # Get API key from config
        api_key = os.environ.get("GEMINI_API_KEY") or AppConfig.GEMINI_API_KEY
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Resolve model name from env/config if not provided by subclass
        self.model_name = self.model_name or AppConfig.MODEL_NAME
        if not self.model_name:
            raise ValueError("MODEL_NAME must be set via environment for GeminiLLM")

        # Initialize model
        self.client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k
            },
            safety_settings=self.safety_settings
        )
        
        logger.info(f"Initialized Gemini LLM with model {self.model_name}")
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type"""
        return "gemini"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        """Generate text using Gemini model"""
        from langchain.schema import Generation
        generations: List[List[Generation]] = []
        for prompt in prompts:
            try:
                response = self.client.generate_content(prompt)
                if getattr(response, "text", None):
                    text = response.text
                else:
                    text = (
                        response.candidates[0].content.parts[0].text
                        if getattr(response, "candidates", None)
                        else ""
                    )
            except Exception as e:
                logger.error(f"Error generating with Gemini: {e}")
                text = ERROR_TEXT
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)
    
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        """Asynchronously generate text using Gemini model"""
        from langchain.schema import Generation
        generations: List[List[Generation]] = []
        for prompt in prompts:
            try:
                response = await self.client.generate_content_async(prompt)
                if getattr(response, "text", None):
                    text = response.text
                else:
                    text = (
                        response.candidates[0].content.parts[0].text
                        if getattr(response, "candidates", None)
                        else ""
                    )
            except Exception as e:
                logger.error(f"Error generating with Gemini: {e}")
                text = ERROR_TEXT
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

class GeminiChatModel(BaseChatModel):
    """
    LangChain adapter for Google Gemini 2.0 API as a chat model
    
    This class implements the LangChain Chat Model interface for Google Gemini 2.0 API.
    """
    
    model_name: str = ""
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 64
    safety_settings: Optional[List[Dict[str, Any]]] = None
    
    def __init__(self, **kwargs):
        """Initialize the Gemini Chat Model"""
        super().__init__(**kwargs)
        
        # Get API key from config
        api_key = os.environ.get("GEMINI_API_KEY") or AppConfig.GEMINI_API_KEY
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Resolve model name from env/config
        self.model_name = self.model_name or AppConfig.MODEL_NAME
        if not self.model_name:
            raise ValueError("MODEL_NAME must be set via environment for GeminiChatModel")

        # Initialize model
        self.client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k
            },
            safety_settings=self.safety_settings
        )
        
        logger.info(f"Initialized Gemini Chat Model with model {self.model_name}")
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type"""
        return "gemini-chat"
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert a list of messages to a single string prompt"""
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"AI: {message.content}")
            else:
                prompt_parts.append(f"{message.type}: {message.content}")
        
        return "\n\n".join(prompt_parts)
    
    def _convert_messages_to_chat(self, messages: List[BaseMessage]) -> genai.ChatSession:
        """Convert messages to a Gemini chat session"""
        # Initialize chat
        chat = self.client.start_chat()
        
        # Add messages to chat
        for message in messages:
            if isinstance(message, SystemMessage):
                # For system messages, we'll add it as a user message with a special prefix
                chat.send_message("SYSTEM INSTRUCTION: " + message.content)
            elif isinstance(message, HumanMessage):
                chat.send_message(message.content)
            elif isinstance(message, AIMessage):
                # For AI messages, we'll skip as they're already part of the history
                pass
        
        return chat
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        """Generate a chat completion using Gemini model"""
        try:
            prompt = self._convert_messages_to_prompt(messages)

            response = self.client.generate_content(prompt)

            if hasattr(response, "text") and response.text:
                text = response.text
            else:
                text = (
                    response.candidates[0].content.parts[0].text
                    if getattr(response, "candidates", None)
                    else ""
                )

            from langchain.schema import Generation
            return LLMResult(generations=[[Generation(text=text)]])

        except Exception as e:
            logger.error(f"Error generating with Gemini Chat: {e}")
            from langchain.schema import Generation
            return LLMResult(generations=[[Generation(text=ERROR_TEXT)]])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        """Asynchronously generate a chat completion using Gemini model"""
        try:
            prompt = self._convert_messages_to_prompt(messages)

            response = await self.client.generate_content_async(prompt)

            if hasattr(response, "text") and response.text:
                text = response.text
            else:
                text = (
                    response.candidates[0].content.parts[0].text
                    if getattr(response, "candidates", None)
                    else ""
                )

            from langchain.schema import Generation
            return LLMResult(generations=[[Generation(text=text)]])

        except Exception as e:
            logger.error(f"Error generating with Gemini Chat: {e}")
            from langchain.schema import Generation
            return LLMResult(generations=[[Generation(text=ERROR_TEXT)]])

def create_gemini_llm(config: Dict[str, Any] = None) -> BaseLanguageModel:
    """
    Create a Gemini LLM instance with the given configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Gemini LLM instance
    """
    config = config or {}
    
    # Determine model type
    model_type = config.get("model_type", "chat")
    
    if model_type == "chat":
        return GeminiChatModel(
            model_name=config.get("model_name") or AppConfig.MODEL_NAME,
            temperature=config.get("temperature", 0.7),
            max_output_tokens=config.get("max_output_tokens", 2048),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k", 64),
            safety_settings=config.get("safety_settings")
        )
    else:
        return GeminiLLM(
            model_name=config.get("model_name") or AppConfig.MODEL_NAME,
            temperature=config.get("temperature", 0.7),
            max_output_tokens=config.get("max_output_tokens", 2048),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k", 64),
            safety_settings=config.get("safety_settings")
        )