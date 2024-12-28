from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from pathlib import Path

class HealthChatModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "RayyanAhmed9477/Health-Chatbot"
        self.local_model_path = Path("fine_tuned_model")
        self.tokenizer = None
        self.model = None

    def load_model(self):
        try:
            # First try loading from local path
            if self.local_model_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.local_model_path),
                    padding_side='left'
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.local_model_path),
                    torch_dtype=torch.float32,
                    device_map=self.device
                )
            else:
                # Fallback to loading from HuggingFace Hub
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    padding_side='left'
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map=self.device
                )
            
            # Ensure the tokenizer has the necessary special tokens
            special_tokens = {
                'pad_token': '[PAD]',
                'eos_token': '</s>',
                'bos_token': '<s>',
                'unk_token': '[UNK]'
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def generate_response(self, prompt, max_length=100):
        if not self.model or not self.tokenizer:
            self.load_model()

        try:
            # Prepare the input prompt
            formatted_prompt = f"User: {prompt}\nAssistant:"
            
            # Tokenize input
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
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2
                )
            
            # Decode and clean up the response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Extract only the assistant's response
            response_parts = response.split("Assistant:")
            if len(response_parts) > 1:
                response = response_parts[1].strip()
            else:
                response = response_parts[0].strip()
            
            # Remove any remaining prompt text
            response = response.replace(prompt, "").strip()
            
            return response

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")