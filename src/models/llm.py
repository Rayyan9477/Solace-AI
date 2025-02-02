from sympy import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from typing import Optional
from config.settings import AppConfig
import logging

logger = logging.getLogger(__name__)

class SafeLLM:
    """Enterprise-grade LLM wrapper with safety features"""
    
    def __init__(self, model_name: str = AppConfig.MODEL_NAME, use_cpu: bool = True):
        self.device = "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._load_model(model_name)
        
        self.generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_new_tokens=AppConfig.MAX_RESPONSE_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.15,
            return_full_text=False
        )
        
    def _load_model(self, model_name: str):
        """Safe model loading with memory management"""
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=False
            ).to(self.device)
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Failed to initialize language model")

    def generate_safe_response(self, prompt: str, context: str = "") -> str:
        """Generate response with safety checks"""
        try:
            full_prompt = f"Context: {context}\nUser: {prompt}\nAssistant:"
            
            output = self.generation_pipeline(
                full_prompt,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )[0]['generated_text']
            
            return self._postprocess(output)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return "I'm having trouble responding right now. Please try again."

    def _postprocess(self, text: str) -> str:
        """Response cleanup and safety filtering"""
        # Remove any incomplete sentences
        text = text.rsplit('.', 1)[0] + '.' if '.' in text else text
        
        # Filter sensitive content
        text = re.sub(r'\b(自杀|自残|self[- ]?harm)\b', '[REDACTED]', text, flags=re.IGNORECASE)
        
        # Trim to last complete sentence
        return text.strip()