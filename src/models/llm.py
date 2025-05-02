from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Callable, ClassVar
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import google.generativeai as genai
from datetime import datetime
import logging
from config.settings import AppConfig
import os

# Only Gemini 2.0 Flash is supported

logger = logging.getLogger(__name__)

class TokenManager:
    """Manages token usage and content filtering"""
    
    def __init__(self):
        self.blocked_patterns = {
            'harmful': ['hack', 'exploit', 'vulnerability'],
            'unsafe': ['password', 'credential', 'secret'],
            'toxic': ['hate', 'slur', 'offensive']
        }
    
    def filter_content(self, text: str, category: str) -> str:
        """Filter content based on category"""
        if category not in self.blocked_patterns:
            return text
            
        filtered_text = text
        for pattern in self.blocked_patterns[category]:
            filtered_text = filtered_text.replace(pattern, '[FILTERED]')
        return filtered_text
    
    def check_toxicity(self, text: str) -> float:
        """Simple toxicity check"""
        toxic_words = set(self.blocked_patterns['toxic'])
        words = set(text.lower().split())
        toxicity_score = len(words.intersection(toxic_words)) / len(words) if words else 0
        return toxicity_score
    
    def get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()

class AgnoLLM(BaseLLM):
    """Custom LLM provider using Google Gemini 2.0 Flash"""
    
    # Add type annotations to class attributes
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2000
    config: Dict[str, Any] = {}
    token_manager: TokenManager = None
    model: Any = None # Declare model field (will hold the genai.GenerativeModel instance)
    
    def __init__(self, model_config: Dict[str, Any] = None, **kwargs):
        """Initialize the LLM with configuration"""
        # Get default config if not provided
        config_data = model_config or AppConfig.LLM_CONFIG
        
        # Validate API key
        if not config_data.get("api_key"):
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Always use AppConfig.GEMINI_API_KEY (set in settings.py)
        config_data["api_key"] = AppConfig.GEMINI_API_KEY
        
        # Set default values from config_data for BaseLLM fields
        kwargs.setdefault("model_name", config_data.get("model", "gemini-2.0-flash"))
        kwargs.setdefault("temperature", config_data.get("temperature", 0.7))
        kwargs.setdefault("top_p", config_data.get("top_p", 0.9))
        kwargs.setdefault("top_k", config_data.get("top_k", 50))
        kwargs.setdefault("max_tokens", config_data.get("max_tokens", 2000))
        
        # Initialize parent class (BaseLLM fields)
        super().__init__(**kwargs)
        
        # Store the full configuration in the declared field
        self.config = config_data
        self.token_manager = TokenManager()
        
        # Initialize Gemini with proper configuration
        try:
            api_key = self.config["api_key"]
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set or is empty")
                
            # Configure Gemini client
            genai.configure(api_key=api_key)
            
            # Configure model with generation settings
            generation_config = genai.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_tokens,
                candidate_count=1
            )
            
            # Set safety settings
            safety_settings = []
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Test the model with a simple generation to verify it works
            test_response = self.model.generate_content("Hello")
            logger.info(f"Model initialization successful: {self.model_name}")
                
        except ValueError as ve:
            logger.error(f"ValueError in model initialization: {str(ve)}")
            self.model = None
            raise ValueError(f"Failed to initialize model: {str(ve)}")
            
        except Exception as e:
            logger.error(f"Error in model initialization: {str(e)}")
            self.model = None
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> List[str]:
        """Generate response synchronously for multiple prompts."""
        results = []
        for prompt in prompts:
            try:
                # Apply safety filters to input
                safe_prompt = self._apply_safety_filters(prompt)
                
                # Call the model
                response = self.model.generate_content(safe_prompt)
                
                # Extract text from response
                text = response.text
                
                # Post-process the response
                processed_text = self._postprocess_response(text)
                
                # Run callback if provided
                if run_manager:
                    run_manager.on_llm_new_token(processed_text)
                    
                # Add to results
                results.append(processed_text)
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                # Provide a fallback response
                results.append("I'm having trouble generating a response right now. Could you try rephrasing your question?")
        
        return results

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> List[str]:
        """Generate response asynchronously for multiple prompts."""
        try:
            import asyncio
            
            # Use a thread pool to run the synchronous _generate method
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self._generate(prompts, stop, run_manager, **kwargs)
            )
            
            return results
        except Exception as e:
            logger.error(f"Error in async generation: {str(e)}")
            return ["I apologize, but I'm unable to process your request at the moment."] * len(prompts)

    def _format_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with context"""
        if not context:
            return prompt
            
        template = """Context: {context}
User Query: {prompt}
Assistant: Let me help you with that."""
        
        return template.format(
            context=context.get('context', ''),
            prompt=prompt
        )
    
    def _apply_safety_filters(self, text: str) -> str:
        """Apply input safety filters"""
        safety_filters = AppConfig.SAFETY_CONFIG.get("blocked_categories", [])
        filtered_text = text
        
        for category in safety_filters:
            filtered_text = self.token_manager.filter_content(filtered_text, category)
            
        return filtered_text
    
    def _postprocess_response(self, text: str) -> str:
        """Clean and validate response"""
        # Remove any incomplete sentences
        text = text.rsplit('.', 1)[0] + '.' if '.' in text else text
        
        # Apply safety filters
        return self._apply_safety_filters(text.strip())
    
    def _check_safety(self, text: str) -> Dict[str, Any]:
        """Check response for safety concerns"""
        flags = {}
        
        # Check toxicity
        toxicity_score = self.token_manager.check_toxicity(text)
        if toxicity_score > AppConfig.SAFETY_CONFIG.get("max_toxicity", 0.7):
            flags["toxicity"] = toxicity_score
            
        # Check for blocked content
        for category in AppConfig.SAFETY_CONFIG.get("blocked_categories", []):
            original_text = text
            filtered_text = self.token_manager.filter_content(text, category)
            
            if original_text != filtered_text:
                flags[category] = True
                
        return flags

    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return "agno-llm"

    @property
    def identifier(self) -> str:
        """A unique identifier for this LLM."""
        return f"AgnoLLM-{self.model_name}"
        
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a response for a given prompt with context
        
        Args:
            prompt: The user's prompt
            context: Optional context information
            
        Returns:
            Dictionary with the response and metadata
        """
        try:
            # Format the prompt with context if provided
            formatted_prompt = self._format_prompt(prompt, context)
            
            # Start timer
            start_time = datetime.now()
            
            # Generate response
            response = self._generate([formatted_prompt])[0]
            
            # End timer
            end_time = datetime.now()
            time_taken = (end_time - start_time).total_seconds()
            
            # Check safety
            safety_flags = self._check_safety(response)
            
            return {
                "response": response,
                "time_taken": time_taken,
                "safety_flags": safety_flags,
                "timestamp": self.token_manager.get_timestamp()
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your request.",
                "error": str(e),
                "timestamp": self.token_manager.get_timestamp()
            }
