"""
LLM module for the Contextual-Chatbot.
Provides language model implementations and wrappers.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output import LLMResult, Generation
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Import Gemini API
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
try:
    from google.generativeai.types.generation_types import StopReason
except ImportError:
    # Define a fallback if the import isn't available
    class StopReason:
        STOP = "STOP"
        MAX_TOKENS = "MAX_TOKENS"
        SAFETY = "SAFETY"
        RECITATION = "RECITATION"
        OTHER = "OTHER"

# Configure logger
logger = logging.getLogger(__name__)

class GeminiLLM(BaseLanguageModel):
    """
    Wrapper for Google's Gemini 2.0 API
    """
    
    def __init__(self, 
                api_key: str,
                model_name: str = "gemini-2.0-pro",
                temperature: float = 0.7,
                max_output_tokens: int = 1024,
                top_p: float = 0.95,
                top_k: int = 40):
        """
        Initialize the Gemini LLM
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name to use
            temperature: Temperature for generation
            max_output_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        super().__init__()
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        self.model_name = model_name
        
        # Create generation config
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k
        )
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=self.generation_config
            )
            logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise
    
    def _generate(self, 
                 prompts: List[str],
                 stop: Optional[List[str]] = None,
                 run_manager: Optional[CallbackManagerForLLMRun] = None) -> LLMResult:
        """
        Generate text completions for the provided prompts
        
        Args:
            prompts: List of prompts to generate from
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            
        Returns:
            LLMResult with generations
        """
        generations = []
        
        for prompt in prompts:
            try:
                # Generate response from Gemini
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=None  # Safety handled by our SafetyAgent
                )
                
                # Process response
                if hasattr(response, 'candidates') and response.candidates:
                    text = response.text
                    
                    # Extract stop reason if available
                    stop_reason = None
                    if hasattr(response.candidates[0], 'finish_reason'):
                        stop_reason = response.candidates[0].finish_reason
                    
                    # Create Generation object
                    gen = Generation(
                        text=text,
                        generation_info={
                            "finish_reason": stop_reason
                        }
                    )
                    generations.append([gen])
                else:
                    # Empty response - return empty generation
                    generations.append([Generation(text="")])
                    
            except Exception as e:
                logger.error(f"Error generating with Gemini: {str(e)}")
                # Return empty generation on error
                generations.append([Generation(text="", generation_info={"error": str(e)})])
        
        return LLMResult(generations=generations)
    
    async def _agenerate(self, 
                       prompts: List[str],
                       stop: Optional[List[str]] = None,
                       run_manager: Optional[CallbackManagerForLLMRun] = None) -> LLMResult:
        """
        Asynchronously generate text completions for the provided prompts
        
        Args:
            prompts: List of prompts to generate from
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            
        Returns:
            LLMResult with generations
        """
        try:
            # For now, we use the synchronous version
            # Gemini Python SDK currently doesn't have full async support
            # This could be updated in the future when supported
            return self._generate(prompts, stop, run_manager)
        except Exception as e:
            logger.error(f"Error in async generation with Gemini: {str(e)}")
            return LLMResult(generations=[[Generation(text="", generation_info={"error": str(e)})]])

    async def agenerate_messages(self, messages, **kwargs):
        """
        Generate text from a list of messages for chat models.
        
        Args:
            messages: List of message objects
            
        Returns:
            LLMResult with generations
        """
        # Convert message format to text prompt compatible with Gemini
        prompt = self._messages_to_prompt(messages)
        
        # Generate using the text prompt
        result = await self._agenerate([prompt], **kwargs)
        return result
    
    def _messages_to_prompt(self, messages) -> str:
        """
        Convert message objects to a text prompt for Gemini
        
        Args:
            messages: List of message objects
            
        Returns:
            Formatted text prompt
        """
        prompt_parts = []
        
        for message in messages:
            # Extract role and content from message
            if hasattr(message, "type") and hasattr(message, "content"):
                role = message.type
                content = message.content
            elif hasattr(message, "role") and hasattr(message, "content"):
                role = message.role
                content = message.content
            else:
                # Try to get content directly
                content = str(message)
                role = "user"
            
            # Format based on role
            if role == "system" or role == "SystemMessage":
                prompt_parts.append(f"System: {content}")
            elif role == "human" or role == "user" or role == "HumanMessage":
                prompt_parts.append(f"User: {content}")
            elif role == "ai" or role == "assistant" or role == "AIMessage":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")
        
        # Join into a single string
        return "\n\n".join(prompt_parts)
    
    def _call(self, 
              prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs) -> str:
        """
        Call the model with a single prompt and return a string response
        
        Args:
            prompt: The prompt to send to the model
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        try:
            # Generate response from Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=None  # Safety handled by our SafetyAgent
            )
            
            # Process response
            if hasattr(response, 'candidates') and response.candidates:
                text = response.text
                return text if text else ""
            else:
                # Empty response
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Gemini: {str(e)}")
            return ""
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return "gemini"

# Factory function to create an LLM based on configuration
def get_llm(config: Dict[str, Any] = None) -> BaseLanguageModel:
    """
    Create a language model instance based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLM instance
    """
    config = config or {}
    
    # Check for Gemini configuration
    if config.get("provider", "").lower() == "gemini":
        # Get API key from config or environment
        api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("Gemini API key not found in config or environment")
        
        # Create Gemini LLM
        return GeminiLLM(
            api_key=api_key,
            model_name=config.get("model_name", "gemini-2.0-pro"),
            temperature=config.get("temperature", 0.7),
            max_output_tokens=config.get("max_output_tokens", 1024),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k", 40)
        )
    
    # Default fallback (should be extended with other providers as needed)
    logger.warning(f"Unknown LLM provider: {config.get('provider')}. Using default.")
    
    # Import a default model (you may want to update this based on your needs)
    from langchain.llms import OpenAI
    return OpenAI()

# AgnoLLM class that wraps LLM for use with Agno framework
class AgnoLLM:
    """LLM wrapper for use with Agno framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration"""
        self.config = config or {}
        
        # Use environment variables for API keys if not in config
        if "api_key" not in self.config:
            # Check for Gemini API key
            if os.getenv("GEMINI_API_KEY"):
                self.config["provider"] = "gemini"
                self.config["api_key"] = os.getenv("GEMINI_API_KEY")
        
        # Initialize the LLM
        self.llm = get_llm(self.config)
    
    async def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response to the prompt
        
        Args:
            prompt: Prompt text
            
        Returns:
            Dictionary with generated response
        """
        try:
            # Generate with LLM
            if hasattr(self.llm, "agenerate"):
                result = await self.llm.agenerate([prompt])
                response = result.generations[0][0].text
            else:
                # Fallback for non-async LLMs
                result = self.llm.generate([prompt])
                response = result.generations[0][0].text
            
            return {
                "response": response,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error generating with LLM: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
