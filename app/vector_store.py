import faiss
import numpy as np
import pickle
import os
import torch

class VectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        
    def add_text(self, text, embedding):
        self.texts.append(text)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        self.index.add(embedding)
        
    def search(self, query_embedding, k=5):
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        distances, indices = self.index.search(query_embedding, k)
        return [(self.texts[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'texts': self.texts, 'index': faiss.serialize_index(self.index)}, f)
    
    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.index = faiss.deserialize_index(data['index'])