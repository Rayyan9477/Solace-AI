"""
Device Utilities for CUDA/CPU Configuration

This module provides centralized device management for the application,
ensuring consistent CUDA usage when available with proper fallback to CPU.
"""

try:
    import torch
except Exception:
    torch = None  # Allow running without torch installed
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

class DeviceManager:
    """Centralized device management for CUDA/CPU operations"""
    
    _instance = None
    _device = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_device()
            self._initialized = True
    
    def _initialize_device(self):
        """Initialize and configure the compute device"""
        try:
            # Check if CUDA is available (when torch is present)
            if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
                # Test CUDA initialization
                try:
                    torch.cuda.init()
                    # Test tensor creation on GPU
                    test_tensor = torch.tensor([1.0], device="cuda")
                    self._device = torch.device("cuda")
                    
                    # Log device information
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
                    
                    logger.info(f"CUDA initialized successfully")
                    logger.info(f"GPU Device: {gpu_name}")
                    logger.info(f"GPU Memory: {memory_total:.2f} GB")
                    logger.info(f"GPU Count: {gpu_count}")
                    
                    # Clean up test tensor
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"CUDA initialization failed, falling back to CPU: {str(e)}")
                    self._device = torch.device("cpu")
            else:
                logger.info("CUDA/torch not available, using CPU")
                self._device = "cpu" if torch is None else torch.device("cpu")
                
        except Exception as e:
            logger.error(f"Error during device initialization: {str(e)}")
            self._device = "cpu" if torch is None else torch.device("cpu")
        
        logger.info(f"Device configured: {self._device}")
    
    @property
    def device(self):
        """Get the configured device"""
        return self._device
    
    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available and being used"""
        try:
            return getattr(self._device, 'type', str(self._device)) == "cuda"
        except (AttributeError, TypeError):
            return False
    
    def get_device_info(self) -> dict:
        """Get detailed device information"""
        info = {
            "device": str(self._device),
            "type": self._device.type,
            "cuda_available": bool(torch and hasattr(torch, 'cuda') and torch.cuda.is_available()),
            "using_cuda": self.is_cuda_available
        }
        
        if self.is_cuda_available and torch is not None:
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
                "gpu_memory_cached": torch.cuda.memory_reserved(0) / 1024**3
            })
        
        return info
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA"""
        if self.is_cuda_available and torch is not None:
            try:
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GPU cache: {str(e)}")
    
    def to_device(self, tensor_or_model):
        """Move tensor or model to the configured device"""
        try:
            return tensor_or_model.to(self._device)
        except Exception as e:
            logger.error(f"Failed to move to device {self._device}: {str(e)}")
            return tensor_or_model


# Global instance
device_manager = DeviceManager()

def get_device() -> torch.device:
    """Get the configured device"""
    return device_manager.device

def is_cuda_available() -> bool:
    """Check if CUDA is available and being used"""
    return device_manager.is_cuda_available

def get_device_info() -> dict:
    """Get detailed device information"""
    return device_manager.get_device_info()

def clear_gpu_cache():
    """Clear GPU cache if using CUDA"""
    device_manager.clear_cache()

def to_device(tensor_or_model):
    """Move tensor or model to the configured device"""
    return device_manager.to_device(tensor_or_model)
