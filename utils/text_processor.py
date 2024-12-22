import torch

class TextProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def encode_text(self, text):
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        return encoded
    
    def decode_text(self, token_ids):
        decoded = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        return decoded
    
    def get_embeddings(self, text, model):
        encoded = self.encode_text(text)
        with torch.no_grad():
            outputs = model(**encoded)
            # Use the last hidden state as embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings