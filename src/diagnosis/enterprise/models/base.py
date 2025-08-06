"""
Base model classes and utilities for neural networks
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    """Base class for all neural network models in the enterprise pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameters count"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config,
            'device': next(self.parameters()).device.type if list(self.parameters()) else 'cpu'
        }
    
    def save_checkpoint(self, filepath: str, additional_info: Optional[Dict[str, Any]] = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'model_class': self.__class__.__name__,
            'model_info': self.get_model_info()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: Optional[str] = None):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model instance
        model = cls(checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device:
            model = model.to(device)
        
        logger.info(f"Model loaded from {filepath}")
        return model, checkpoint.get('model_info', {})

class ModelRegistry:
    """Registry for managing multiple models"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
    
    def register_model(self, name: str, model: BaseModel, config: Dict[str, Any]):
        """Register a model with the registry"""
        self.models[name] = model
        self.model_configs[name] = config
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get a model by name"""
        return self.models.get(name)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models with their info"""
        return {
            name: model.get_model_info() 
            for name, model in self.models.items()
        }
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from registry"""
        if name in self.models:
            del self.models[name]
            del self.model_configs[name]
            logger.info(f"Removed model: {name}")
            return True
        return False

def initialize_weights(module: nn.Module):
    """Initialize model weights using best practices"""
    if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        # Standard initialization for layer norm
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        # Initialize RNN weights
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def freeze_layers(model: nn.Module, layer_names: list):
    """Freeze specific layers in a model"""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            logger.info(f"Frozen layer: {name}")

def unfreeze_layers(model: nn.Module, layer_names: list):
    """Unfreeze specific layers in a model"""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            logger.info(f"Unfrozen layer: {name}")

class ModelUtils:
    """Utility functions for model management"""
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get information about available devices"""
        info = {
            'cpu_available': True,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if info['cuda_available']:
            info['cuda_devices'] = [
                {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory': torch.cuda.get_device_properties(i).total_memory
                }
                for i in range(torch.cuda.device_count())
            ]
        
        return info
    
    @staticmethod
    def select_best_device() -> str:
        """Select the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @staticmethod
    def get_memory_usage(device: str = "cuda") -> Dict[str, int]:
        """Get memory usage for a device"""
        if device == "cuda" and torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        else:
            return {'allocated': 0, 'cached': 0, 'max_allocated': 0}
    
    @staticmethod
    def clear_memory(device: str = "cuda"):
        """Clear memory cache"""
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")

# Global model registry instance
model_registry = ModelRegistry()