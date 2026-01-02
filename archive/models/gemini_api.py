"""
Gemini API wrapper for the Contextual-Chatbot.
Provides an interface to Google's Gemini 2.0 API.
"""

import os
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import google.generativeai as genai
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import LLMResult, Generation, AIMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

# Configure logger
logger = logging.getLogger(__name__)
from src.config.settings import AppConfig

class GeminiAPI(LLM, BaseLanguageModel):
    """
    Wrapper for Google's Gemini 2.0 API
    Implements the LangChain BaseLanguageModel interface
    """

    # No hardcoded default; must be provided via env/config
    model_name: str = ""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 1024
    api_key: Optional[str] = None
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, **kwargs):
        """
        Initialize the Gemini API wrapper
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model name/version
            **kwargs: Additional parameters for the LLM
        """
        kwargs.pop("model_kwargs", None)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or AppConfig.MODEL_NAME
        if not self.model_name:
            raise ValueError("MODEL_NAME must be set via environment or passed to GeminiAPI")

        # Extract parameters from kwargs or use defaults
        self.temperature = kwargs.pop("temperature", self.temperature)
        self.top_p = kwargs.pop("top_p", self.top_p)
        self.top_k = kwargs.pop("top_k", self.top_k)
        self.max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        # Initialize Gemini
        genai.configure(api_key=self.api_key)

        # Configure the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_tokens,
            }
        )

        # Initialize LLM base class
        super().__init__(**kwargs)

        logger.info("Initialized Gemini API with model: %s", self.model_name)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """
        Call the Gemini API with the given prompt
        
        Args:
            prompt: The prompt to send to Gemini
            stop: Sequences that trigger early stopping
            run_manager: Callback manager
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        try:
            # Configure request parameters
            request_params = {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "top_k": kwargs.get("top_k", self.top_k),
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            # Call the Gemini API
            response = self.model.generate_content(
                prompt,
                generation_config=request_params,
                safety_settings=kwargs.get("safety_settings", None)
            )
            
            # Handle empty response
            if not response.text:
                logger.warning("Gemini returned an empty response")
                return "I'm not sure how to respond to that."
                
            return response.text
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    async def _agenerate(
        self, 
        prompts: List[str], 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs
    ) -> LLMResult:
        """
        Asynchronously generate text from prompts
        
        Args:
            prompts: List of prompts to process
            stop: Sequences that trigger early stopping
            run_manager: Callback manager
            **kwargs: Additional parameters
            
        Returns:
            LLMResult with generated texts
        """
        generations = []
        
        for prompt in prompts:
            try:
                # Configure request parameters
                request_params = {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "top_p": kwargs.get("top_p", self.top_p),
                    "top_k": kwargs.get("top_k", self.top_k),
                    "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                
                # Call the Gemini API
                response = self.model.generate_content(
                    prompt,
                    generation_config=request_params,
                    safety_settings=kwargs.get("safety_settings", None)
                )
                
                # Process the response
                text = response.text if response.text else "I'm not sure how to respond to that."
                
                # Create a Generation object
                gen = Generation(text=text)
                generations.append([gen])
                
            except Exception as e:
                logger.error(f"Error in async Gemini generation: {str(e)}")
                # Add a fallback generation
                gen = Generation(text="I'm having trouble processing your request right now. Please try again later.")
                generations.append([gen])
        
        # Create and return an LLMResult
        return LLMResult(generations=generations)
    
    async def agenerate_messages(
        self, 
        messages: List[Any], 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs
    ) -> LLMResult:
        """
        Asynchronously generate text from message objects
        
        Args:
            messages: List of message objects to process
            stop: Sequences that trigger early stopping
            run_manager: Callback manager
            **kwargs: Additional parameters
            
        Returns:
            LLMResult with generated texts
        """
        # Extract text from messages
        prompts = []
        for message in messages:
            if hasattr(message, "content"):
                prompts.append(message.content)
            else:
                # Try to convert message to string
                prompts.append(str(message))
        
        # Call the agenerate method
        return await self._agenerate(prompts, stop, run_manager, **kwargs)
    
    def enhance_prompt_for_empathy(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Enhance a prompt with contextual information to increase empathy
        
        Args:
            prompt: Base prompt
            context: Contextual information including emotion and diagnosis data
            
        Returns:
            Enhanced prompt
        """
        context = context or {}
        
        # Add emotional context if available
        emotion_prompt = ""
        if "emotion" in context and context["emotion"]:
            emotion_data = context["emotion"]
            primary_emotion = emotion_data.get("primary_emotion", "neutral")
            intensity = emotion_data.get("intensity", 5)
            emotion_prompt = f"\nThe user appears to be feeling {primary_emotion} with an intensity of {intensity}/10."
            
            # Add secondary emotions if available
            secondary_emotions = emotion_data.get("secondary_emotions", [])
            if secondary_emotions:
                emotion_prompt += f" They may also be experiencing {', '.join(secondary_emotions)}."
        
        # Add diagnosis context if available
        diagnosis_prompt = ""
        if "diagnosis" in context and context["diagnosis"]:
            diagnosis_data = context["diagnosis"]
            if isinstance(diagnosis_data, str):
                diagnosis_prompt = f"\nRelevant diagnosis information: {diagnosis_data}"
            else:
                # Try to extract structured diagnosis data
                conditions = diagnosis_data.get("conditions", [])
                if conditions:
                    condition_names = [c.get("name", "unknown") for c in conditions]
                    diagnosis_prompt = f"\nThe user may be experiencing: {', '.join(condition_names)}."
                    
                    # Add severity if available
                    severity = diagnosis_data.get("severity", "")
                    if severity:
                        diagnosis_prompt += f" Severity level: {severity}."
        
        # Add cultural context if available
        cultural_prompt = ""
        if "culture" in context and context["culture"]:
            cultural_data = context["culture"]
            cultural_prompt = f"\nBe sensitive to the user's cultural background: {cultural_data}"
        
        # Combine all prompts with empathy instructions
        empathy_instructions = """
Please provide a response that is:
1. Empathetic and validates the user's feelings
2. Non-judgmental and supportive
3. Culturally sensitive and respectful
4. Focused on understanding rather than giving immediate solutions
5. Tailored to the user's emotional state and needs
"""
        
        enhanced_prompt = f"{prompt}{emotion_prompt}{diagnosis_prompt}{cultural_prompt}\n{empathy_instructions}"
        return enhanced_prompt
    
    def generate_empathetic_response(
        self, 
        message: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate an empathetic response to a user message
        
        Args:
            message: User message
            context: Additional context including emotion and diagnosis data
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Enhance prompt with context for empathy
            enhanced_prompt = self.enhance_prompt_for_empathy(message, context)
            
            # Generate response
            response_text = self._call(enhanced_prompt)
            
            return {
                "response": response_text,
                "prompt": enhanced_prompt,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating empathetic response: {str(e)}")
            return {
                "response": "I'm having difficulty understanding right now. Could we try a different approach?",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def __repr__(self) -> str:
        """String representation of the Gemini API"""
        return f"GeminiAPI(model_name={self.model_name}, temperature={self.temperature})"