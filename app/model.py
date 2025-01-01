from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch
from pathlib import Path
import numpy as np


class HealthChatModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "RayyanAhmed9477/Health-Chatbot"
        self.local_model_path = Path("fine_tuned_model")
        self.tokenizer = None
        self.model = None

    def load_model(self):
        try:
            # First try loading the tokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.model_name,
                padding_side='left'
            )
            
            # Set special tokens
            special_tokens_dict = {
                'pad_token': '[PAD]',
                'bos_token': '<s>',
                'eos_token': '</s>',
                'unk_token': '[UNK]'
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)

            # Load the base model
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )

            # Resize token embeddings using the averaging method
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Initialize new embeddings using the averaging method
            self._initialize_new_embeddings()
            
            # Move model to device
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _initialize_new_embeddings(self):
        """Initialize new embeddings using the averaging method from the paper"""
        embeddings = self.model.get_input_embeddings().weight.data
        num_old_tokens = len(self.tokenizer) - len(self.tokenizer.get_added_vocab())
        
        if num_old_tokens < len(self.tokenizer):
            # Calculate mean and covariance of existing embeddings
            old_embeddings = embeddings[:num_old_tokens].cpu().numpy()
            mu = np.mean(old_embeddings, axis=0)
            sigma = np.cov(old_embeddings.T)
            
            # Generate new embeddings
            num_new_tokens = len(self.tokenizer) - num_old_tokens
            new_embeddings = np.random.multivariate_normal(
                mu, sigma * 1e-5, size=num_new_tokens
            )
            
            # Update the embeddings
            embeddings[num_old_tokens:] = torch.tensor(
                new_embeddings, 
                dtype=embeddings.dtype,
                device=embeddings.device
            )
            
            # Update both input and output embeddings
            self.model.get_input_embeddings().weight.data = embeddings
            if hasattr(self.model, 'lm_head'):
                self.model.lm_head.weight.data = embeddings.clone()

    def generate_response(self, prompt, max_length=100):
        if not self.model or not self.tokenizer:
            self.load_model()

        try:
            # Format the prompt
            formatted_prompt = f"User: {prompt}\nAssistant:"
            
            # Tokenize with proper padding
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length + inputs['input_ids'].shape[1],
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Extract assistant's response
            response_parts = response.split("Assistant:")
            if len(response_parts) > 1:
                response = response_parts[1].strip()
            else:
                response = response_parts[0].replace(prompt, "").strip()
            
            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."