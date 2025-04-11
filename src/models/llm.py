from typing import Optional, List, Dict, Any
from langchain.llms import BaseLLM
import google.generativeai as genai
from datetime import datetime
import logging
from config.settings import AppConfig
import os
import json

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
    """Custom LLM provider using Google Gemini 2.5 Pro"""
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__()
        self.config = model_config or AppConfig.LLM_CONFIG
        self.token_manager = TokenManager()
        
        # Initialize Gemini with proper configuration
        try:
            genai.configure(api_key=self.config["api_key"])
            
            # Configure model with experimental settings
            generation_config = genai.types.GenerationConfig(
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config.get("top_k", 50),
                max_output_tokens=self.config["max_tokens"],
                candidate_count=1
            )
            
            safety_settings = [
                {
                    "category": genai.types.HarmCategory.HARASSMENT,
                    "threshold": genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE
                },
                {
                    "category": genai.types.HarmCategory.HATE_SPEECH,
                    "threshold": genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE
                },
                {
                    "category": genai.types.HarmCategory.DANGEROUS_CONTENT,
                    "threshold": genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE
                },
                {
                    "category": genai.types.HarmCategory.SEXUALLY_EXPLICIT,
                    "threshold": genai.types.HarmBlockThreshold.MEDIUM_AND_ABOVE
                }
            ]
            
            self.model = genai.GenerativeModel(
                model_name=self.config["model"],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            logger.info(f"Successfully initialized {self.config['model']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            self.model = None
    
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response with safety checks and metadata"""
        try:
            if not self.model:
                return {
                    'response': "I apologize, but I'm having trouble generating a response right now.",
                    'error': "Model not initialized",
                    'metadata': {
                        'model': self.config["model"],
                        'timestamp': self.token_manager.get_timestamp(),
                        'fallback': True
                    }
                }
            
            # Apply safety filters to prompt
            safe_prompt = self._apply_safety_filters(prompt)
            
            # Format prompt with context
            formatted_prompt = self._format_prompt(safe_prompt, context)
            
            try:
                # Generate response
                response = self.model.generate_content(
                    formatted_prompt,
                    generation_config={
                        **self.config["generation_config"],
                        **kwargs
                    }
                )
                
                # Handle potential blocking
                if not response.text:
                    return {
                        'response': "I apologize, but I cannot generate that type of content.",
                        'metadata': {
                            'model': self.config["model"],
                            'safety_flags': {'blocked': True},
                            'timestamp': self.token_manager.get_timestamp()
                        }
                    }
                
                # Process and validate response
                processed_response = self._postprocess_response(response.text)
                
                return {
                    'response': processed_response,
                    'metadata': {
                        'model': self.config["model"],
                        'safety_flags': self._check_safety(processed_response),
                        'timestamp': self.token_manager.get_timestamp()
                    }
                }
                
            except Exception as gen_error:
                logger.error(f"Generation error: {str(gen_error)}")
                # Try again with more conservative settings
                try:
                    response = self.model.generate_content(
                        formatted_prompt,
                        generation_config={
                            "temperature": 0.5,
                            "top_p": 0.8,
                            "top_k": 40,
                            "max_output_tokens": min(1000, self.config["max_tokens"])
                        }
                    )
                    processed_response = self._postprocess_response(response.text)
                    
                    return {
                        'response': processed_response,
                        'metadata': {
                            'model': self.config["model"],
                            'safety_flags': self._check_safety(processed_response),
                            'timestamp': self.token_manager.get_timestamp(),
                            'fallback_settings': True
                        }
                    }
                except Exception as fallback_error:
                    logger.error(f"Fallback generation failed: {str(fallback_error)}")
                    raise
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                'response': "I apologize, but I'm having trouble processing your request right now.",
                'error': str(e),
                'metadata': {
                    'model': self.config["model"],
                    'timestamp': self.token_manager.get_timestamp(),
                    'fallback': True
                }
            }
    
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

    @property
    def identifier(self) -> str:
        """Get model identifier"""
        return f"gemini-{self.config['model']}"