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
                raise ValueError("API key is empty or not set")
                
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
            if test_response:
                logger.info(f"Successfully initialized {self.model_name}")
            else:
                raise ValueError("Model initialization test failed")
                
        except ValueError as ve:
            logger.error(f"Configuration error: {str(ve)}")
            raise ValueError(f"Failed to initialize Gemini model: {str(ve)}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise RuntimeError(f"Unexpected error initializing Gemini model: {str(e)}")
    
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
                if not self.model:
                    response_text = "I apologize, but I'm having trouble generating a response right now. Model not initialized."
                else:
                    safe_prompt = self._apply_safety_filters(prompt)
                    formatted_prompt = self._format_prompt(safe_prompt, None)
                    
                    try:
                        generation_config = self.config.get("generation_config", {})
                        response = self.model.generate_content(
                            formatted_prompt,
                            generation_config=generation_config,
                        )
                        
                        if not response.parts:
                            block_reason = response.prompt_feedback.block_reason if hasattr(response.prompt_feedback, 'block_reason') else 'Unknown'
                            logger.warning(f"Generation blocked. Reason: {block_reason}")
                            response_text = "I apologize, but I cannot generate that type of content."
                        elif not response.text:
                            logger.warning(f"Generation returned empty text. Parts: {response.parts}, Candidates: {response.candidates}")
                            response_text = "I received an empty response from the model."
                        else:
                            processed_response = self._postprocess_response(response.text)
                            safety_flags = self._check_safety(processed_response)
                            if safety_flags:
                                logger.warning(f"Response flagged for safety concerns: {safety_flags}")
                            response_text = processed_response
                            
                    except Exception as gen_error:
                        logger.error(f"Generation error: {str(gen_error)}")
                        response_text = "I apologize, but I encountered an error during generation."
                
                results.append(response_text)

            except Exception as e:
                logger.error(f"Outer generation loop failed for prompt: {prompt[:50]}... Error: {str(e)}")
                results.append("I apologize, but I'm having trouble processing your request right now.")
        
        return results

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        logger.warning("Using synchronous _generate within _agenerate. Consider implementing true async call.")
        return self._generate(prompts, stop, **kwargs)

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
            filtered_text = self.token_manager.filter_content(
                filtered_text,
                category
            )
            
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
        if self.token_manager.check_toxicity(text) > AppConfig.SAFETY_CONFIG.get("max_toxicity", 0.7):
            flags['high_toxicity'] = True
            
        # Check for blocked content
        for category in AppConfig.SAFETY_CONFIG.get("blocked_categories", []):
            if any(pattern in text.lower() for pattern in self.token_manager.blocked_patterns.get(category, [])):
                flags[f'contains_{category}'] = True
                
        return flags

    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return "agno-llm"

    @property
    def identifier(self) -> str:
        """Get model identifier"""
        return f"gemini-{self.model_name}"
