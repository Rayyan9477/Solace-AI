from typing import Optional, List, Dict, Any
from langchain.llms import BaseLLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from datetime import datetime
import logging
from config.settings import AppConfig

logger = logging.getLogger(__name__)

class TokenManager:
    """Manages token usage and content filtering"""
    
    def __init__(self):
        self.blocked_patterns = {
            'harmful': ['hack', 'exploit', 'vulnerability'],
            'unsafe': ['password', 'credential', 'secret'],
            'toxic': ['hate', 'slur', 'offensive']
        }
    
    def calculate_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Calculate token usage for prompt and response"""
        return {
            'prompt_tokens': len(prompt.split()),
            'response_tokens': len(response.split()),
            'total_tokens': len(prompt.split()) + len(response.split())
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
    
    def contains_blocked_content(self, text: str, category: str) -> bool:
        """Check if text contains blocked content"""
        if category not in self.blocked_patterns:
            return False
            
        return any(pattern in text.lower() for pattern in self.blocked_patterns[category])
    
    def get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self):
        self.token_manager = TokenManager()
    
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response with safety checks and metadata"""
        raise NotImplementedError("Subclasses must implement generate()")

class AgnoLLM(LLMProvider):
    """Custom LLM provider with safety features"""
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__()
        self.config = model_config or AppConfig.LLM_CONFIG
        self.device = "cuda" if torch.cuda.is_available() and not self.config.get("use_cpu", True) else "cpu"
        
        # Initialize model and tokenizer with error handling
        try:
            self.tokenizer = self._load_tokenizer()
            self.model = self._load_model()
            self.pipeline = self._create_pipeline()
        except Exception as e:
            logger.error(f"Failed to initialize AgnoLLM: {str(e)}")
            # Set fallback values to prevent errors
            self.tokenizer = None
            self.model = None
            self.pipeline = None
            
    def _load_tokenizer(self):
        """Load tokenizer with safety checks"""
        try:
            return AutoTokenizer.from_pretrained(
                self.config["model"],
                trust_remote_code=True,  # Add trust_remote_code parameter
                padding_side="left"  # Add padding_side parameter
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            # Try with a fallback model if the primary one fails
            try:
                logger.warning("Attempting to load fallback tokenizer")
                return AutoTokenizer.from_pretrained(
                    "gpt2",  # Use a simple fallback model
                    trust_remote_code=True
                )
            except Exception as fallback_error:
                logger.error(f"Fallback tokenizer also failed: {str(fallback_error)}")
                raise RuntimeError("Tokenizer initialization failed")
            
    def _load_model(self):
        """Load model with memory optimization"""
        try:
            # Use a try-except block to handle potential PyTorch class instantiation errors
            try:
                return AutoModelForCausalLM.from_pretrained(
                    self.config["model"],
                    device_map="auto",  # Use auto instead of specific device
                    torch_dtype=torch.float32,  # Use float32 for better compatibility
                    low_cpu_mem_usage=True,
                    trust_remote_code=True  # Add trust_remote_code parameter
                )
            except RuntimeError as e:
                if "Tried to instantiate class" in str(e):
                    # Handle the specific PyTorch class instantiation error
                    logger.warning(f"PyTorch class instantiation error: {str(e)}")
                    # Try an alternative approach with more conservative settings
                    return AutoModelForCausalLM.from_pretrained(
                        self.config["model"],
                        device_map="cpu",  # Force CPU
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError("Model initialization failed")
            
    def _create_pipeline(self):
        """Create optimized generation pipeline"""
        try:
            return pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=self.config.get("max_tokens", 2000),
                do_sample=True,
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                repetition_penalty=1.15,
                return_full_text=False
            )
        except Exception as e:
            logger.error(f"Failed to create pipeline: {str(e)}")
            # Try with more conservative settings
            try:
                logger.warning("Attempting to create pipeline with conservative settings")
                return pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # Force CPU
                    max_new_tokens=1000,  # Reduce max tokens
                    do_sample=False,  # Disable sampling
                    temperature=0.5,  # Lower temperature
                    top_p=0.8,
                    repetition_penalty=1.0,
                    return_full_text=False
                )
            except Exception as fallback_error:
                logger.error(f"Fallback pipeline also failed: {str(fallback_error)}")
                raise RuntimeError("Pipeline creation failed")
            
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response with safety checks and metadata"""
        try:
            # Check if pipeline is available
            if not self.pipeline:
                logger.warning("Pipeline not available, using fallback response")
                return {
                    'response': "I apologize, but I'm having trouble generating a response right now. Please try again later.",
                    'metadata': {
                        'model': self.config["model"],
                        'token_usage': {'prompt_tokens': 0, 'response_tokens': 0, 'total_tokens': 0},
                        'safety_flags': {},
                        'timestamp': self.token_manager.get_timestamp(),
                        'fallback': True
                    }
                }
                
            # Apply safety filters to prompt
            safe_prompt = self._apply_safety_filters(prompt)
            
            # Format prompt with context
            formatted_prompt = self._format_prompt(safe_prompt, context)
            
            # Generate response
            output = self.pipeline(
                formatted_prompt,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                **kwargs
            )[0]['generated_text']
            
            # Post-process and validate response
            processed_response = self._postprocess_response(output)
            
            # Calculate token usage
            token_usage = self.token_manager.calculate_usage(
                prompt=formatted_prompt,
                response=processed_response
            )
            
            return {
                'response': processed_response,
                'metadata': {
                    'model': self.config["model"],
                    'token_usage': token_usage,
                    'safety_flags': self._check_safety(processed_response),
                    'timestamp': self.token_manager.get_timestamp()
                }
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                'response': "I apologize, but I'm having trouble generating a response right now.",
                'error': str(e)
            }
            
    def _format_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with context"""
        if not context:
            return prompt
            
        template = """Context: {context}
User Query: {prompt}
Assistant Response:"""
        
        return template.format(
            context=context.get('context', ''),
            prompt=prompt
        )
        
    def _apply_safety_filters(self, text: str) -> str:
        """Apply input safety filters"""
        # Remove potentially harmful patterns
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
            if self.token_manager.contains_blocked_content(text, category):
                flags[f'contains_{category}'] = True
                
        return flags
        
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.config["model"],
            'device': self.device,
            'max_tokens': self.config.get("max_tokens", 2000),
            'temperature': self.config.get("temperature", 0.7)
        }