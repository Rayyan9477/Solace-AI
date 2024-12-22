from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class ModelHandler:
    def __init__(self, model_name="sujal011/llama3.2-3b-mental-health-chatbot"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model_and_tokenizer(self, token=None):
        try:
            # Load primary model and tokenizer with token
            model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=token)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=token)
            
            # Create text-generation pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=self.device
            )
            return generator
        except OSError as e:
            print(f"Warning: {e}. Attempting to load fallback local model.")
            try:
                model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")
                tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
                
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device
                )
                return generator
            except Exception as fallback_e:
                print(f"Error loading fallback model: {fallback_e}")
                raise Exception("Failed to load both primary and fallback models.")