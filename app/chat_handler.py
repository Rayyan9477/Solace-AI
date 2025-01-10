from typing import List, Dict
import torch
from app.vector_store import VectorStore
from transformers import AutoModel, AutoTokenizer

class ChatHandler:
    def __init__(self, model_name="RayyanAhmed9477/CPU-Compatible-Mental-Health-Model", config=None):
        # Initialize model and tokenizer
        try:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except OSError as e:
            # Fallback to local fine_tuned_model if remote isn't found
            print(f"Warning: {e}. Falling back to local fine_tuned_model.")
            self.model = AutoModel.from_pretrained("fine_tuned_model")
            self.tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
        self.config = config
        self.vector_store = VectorStore(config['vector_store']['dimension'])
        
    def get_response(self, user_input: str) -> str:
        try:
            # Generate response using pipeline
            response = self.generator(
                user_input,
                max_length=self.config['model']['max_length'],
                temperature=self.config['model']['temperature'],
                top_p=self.config['model']['top_p'],
                top_k=self.config['model']['top_k'],
                repetition_penalty=self.config['model']['repetition_penalty'],
                do_sample=True
            )[0]['generated_text']
            
            # Clean response
            response = response.replace(user_input, "").strip()
            
            # Update conversation history
            self.conversation_history.append({
                "user": user_input,
                "assistant": response
            })
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _get_conversation_context(self):
        context = self.config['chat']['system_prompt'] + "\n\n"
        for message in self.conversation_history[-3:]:  # Get last 3 messages for context
            context += f"User: {message['user']}\nAssistant: {message['assistant']}\n"
        return context
    
    def _update_history(self, user_input: str, response: str):
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        if len(self.conversation_history) > self.config['chat']['max_context_length']:
            self.conversation_history.pop(0)