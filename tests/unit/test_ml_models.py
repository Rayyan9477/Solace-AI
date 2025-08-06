"""
Unit tests for ML models
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import logging

from src.ml_models.base import BaseModel, ModelRegistry, initialize_weights
from src.ml_models.bayesian import BayesianDiagnosticLayer, UncertaintyQuantifier
from src.ml_models.fusion import MultiModalAttention, AdaptiveFusion

logger = logging.getLogger(__name__)

class TestBaseModel:
    """Test cases for BaseModel"""
    
    def test_base_model_abstract(self):
        """Test that BaseModel cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseModel({})
    
    def test_model_registry(self):
        """Test model registry functionality"""
        registry = ModelRegistry()
        
        # Create a mock model
        mock_model = Mock(spec=BaseModel)
        mock_model.get_model_info.return_value = {
            "model_name": "test_model",
            "parameters": 1000
        }
        
        # Test registration
        registry.register_model("test_model", mock_model, {"test": True})
        assert "test_model" in registry.models
        
        # Test retrieval
        retrieved_model = registry.get_model("test_model")
        assert retrieved_model == mock_model
        
        # Test listing
        models_list = registry.list_models()
        assert "test_model" in models_list
        
        # Test removal
        success = registry.remove_model("test_model")
        assert success
        assert "test_model" not in registry.models

    def test_initialize_weights(self):
        """Test weight initialization function"""
        # Test Linear layer initialization
        linear_layer = torch.nn.Linear(10, 5)
        initialize_weights(linear_layer)
        
        # Check that weights are not all zeros
        assert not torch.allclose(linear_layer.weight, torch.zeros_like(linear_layer.weight))
        
        # Test LayerNorm initialization
        layer_norm = torch.nn.LayerNorm(10)
        initialize_weights(layer_norm)
        
        # LayerNorm should have weight=1 and bias=0
        assert torch.allclose(layer_norm.weight, torch.ones_like(layer_norm.weight))
        assert torch.allclose(layer_norm.bias, torch.zeros_like(layer_norm.bias))

class TestBayesianDiagnosticLayer:
    """Test cases for Bayesian diagnostic layer"""
    
    @pytest.fixture
    def bayesian_layer(self):
        """Create a Bayesian diagnostic layer for testing"""
        config = {
            'input_dim': 512,
            'output_dim': 5,
            'prior_std': 0.1
        }
        return BayesianDiagnosticLayer(config)
    
    def test_initialization(self, bayesian_layer):
        """Test Bayesian layer initialization"""
        assert bayesian_layer.input_dim == 512
        assert bayesian_layer.output_dim == 5
        assert bayesian_layer.prior_std == 0.1
        
        # Check parameter shapes
        assert bayesian_layer.weight_mean.shape == (5, 512)
        assert bayesian_layer.bias_mean.shape == (5,)
    
    def test_forward_pass(self, bayesian_layer):
        """Test forward pass of Bayesian layer"""
        batch_size = 8
        input_tensor = torch.randn(batch_size, 512)
        
        # Test deterministic forward pass
        output = bayesian_layer(input_tensor, sample=False)
        assert output.shape == (batch_size, 5)
        
        # Test stochastic forward pass
        output_stochastic = bayesian_layer(input_tensor, sample=True)
        assert output_stochastic.shape == (batch_size, 5)
        
        # Test multiple samples
        output_samples = bayesian_layer(input_tensor, sample=True, num_samples=10)
        assert output_samples.shape == (10, batch_size, 5)
    
    def test_kl_divergence(self, bayesian_layer):
        """Test KL divergence computation"""
        kl_div = bayesian_layer.kl_divergence()
        
        # KL divergence should be a scalar
        assert kl_div.ndim == 0
        # Should be non-negative
        assert kl_div >= 0
    
    def test_uncertainty_metrics(self, bayesian_layer):
        """Test uncertainty quantification metrics"""
        uncertainty_info = bayesian_layer.get_weight_uncertainty()
        
        expected_keys = ['weight_mean', 'weight_std', 'bias_mean', 'bias_std', 
                        'weight_uncertainty', 'bias_uncertainty']
        
        for key in expected_keys:
            assert key in uncertainty_info

class TestMultiModalAttention:
    """Test cases for multi-modal attention"""
    
    @pytest.fixture
    def attention_model(self):
        """Create multi-modal attention model for testing"""
        config = {
            'fusion_dim': 256,
            'attention_heads': 8,
            'dropout': 0.1
        }
        return MultiModalAttention(config)
    
    def test_initialization(self, attention_model):
        """Test multi-modal attention initialization"""
        assert attention_model.d_model == 256
        assert attention_model.n_heads == 8
        assert attention_model.dropout == 0.1
        
        # Check modality projections exist
        expected_modalities = ['text', 'voice', 'behavioral', 'temporal', 'physiological', 'contextual']
        for modality in expected_modalities:
            assert modality in attention_model.modality_projections
    
    def test_forward_pass(self, attention_model):
        """Test forward pass with multiple modalities"""
        batch_size = 4
        
        # Create mock features for different modalities
        modality_features = {
            'text': torch.randn(batch_size, 768),
            'voice': torch.randn(batch_size, 1024),
            'behavioral': torch.randn(batch_size, 256)
        }
        
        output = attention_model(modality_features)
        assert output.shape == (batch_size, 256)
    
    def test_empty_features_error(self, attention_model):
        """Test error handling for empty features"""
        with pytest.raises(ValueError, match="No modality features provided"):
            attention_model({})
    
    def test_invalid_modality_handling(self, attention_model):
        """Test handling of invalid modalities"""
        batch_size = 4
        
        # Include both valid and invalid modalities
        modality_features = {
            'text': torch.randn(batch_size, 768),
            'invalid_modality': torch.randn(batch_size, 100)
        }
        
        # Should work with valid modalities only
        output = attention_model(modality_features)
        assert output.shape == (batch_size, 256)
    
    def test_attention_weights(self, attention_model):
        """Test attention weights extraction"""
        batch_size = 4
        modality_features = {
            'text': torch.randn(batch_size, 768),
            'voice': torch.randn(batch_size, 1024)
        }
        
        attention_weights = attention_model.get_attention_weights(modality_features)
        
        # Should return attention weights
        assert attention_weights is not None
        assert attention_weights.shape[0] == batch_size

class TestAdaptiveFusion:
    """Test cases for adaptive fusion"""
    
    @pytest.fixture
    def fusion_model(self):
        """Create adaptive fusion model for testing"""
        config = {
            'fusion_dim': 256,
            'max_modalities': 6,
            'dropout': 0.1
        }
        return AdaptiveFusion(config)
    
    def test_initialization(self, fusion_model):
        """Test adaptive fusion initialization"""
        assert fusion_model.d_model == 256
        assert fusion_model.num_modalities == 6
        
        # Check learned parameters
        assert fusion_model.modality_importance.shape == (6,)
    
    def test_forward_pass(self, fusion_model):
        """Test forward pass of adaptive fusion"""
        batch_size = 4
        num_modalities = 3
        
        # Create stacked modality features
        modality_features = torch.randn(batch_size, num_modalities, 256)
        
        output = fusion_model(modality_features)
        assert output.shape == (batch_size, 256)
    
    def test_masked_fusion(self, fusion_model):
        """Test fusion with modality mask"""
        batch_size = 4
        num_modalities = 3
        
        modality_features = torch.randn(batch_size, num_modalities, 256)
        modality_mask = torch.tensor([1.0, 0.0, 1.0])  # Mask out second modality
        
        output = fusion_model(modality_features, modality_mask)
        assert output.shape == (batch_size, 256)

class TestUncertaintyQuantifier:
    """Test cases for uncertainty quantification utilities"""
    
    def test_monte_carlo_predictions(self):
        """Test Monte Carlo uncertainty estimation"""
        # Create a mock Bayesian model
        config = {'input_dim': 10, 'output_dim': 3}
        model = BayesianDiagnosticLayer(config)
        
        input_tensor = torch.randn(5, 10)
        
        # Test Monte Carlo predictions
        results = UncertaintyQuantifier.monte_carlo_predictions(
            model, input_tensor, num_samples=50
        )
        
        expected_keys = ['mean_logits', 'std_logits', 'mean_probs', 'std_probs',
                        'predictive_entropy', 'mutual_information', 'epistemic_uncertainty',
                        'aleatoric_uncertainty', 'all_predictions']
        
        for key in expected_keys:
            assert key in results
        
        # Check shapes
        assert results['mean_logits'].shape == (5, 3)
        assert results['all_predictions'].shape == (50, 5, 3)
    
    def test_calibration_error(self):
        """Test calibration error calculation"""
        # Create mock predictions and targets
        predictions = torch.softmax(torch.randn(100, 5), dim=1)
        targets = torch.randint(0, 5, (100,))
        
        calibration_error = UncertaintyQuantifier.calibration_error(
            predictions, targets, num_bins=10
        )
        
        # Calibration error should be between 0 and 1
        assert 0 <= calibration_error <= 1
    
    def test_prediction_intervals(self):
        """Test prediction interval calculation"""
        # Create mock predictions from multiple samples
        predictions = torch.randn(100, 20, 5)  # 100 samples, 20 batch, 5 outputs
        
        lower_bound, upper_bound = UncertaintyQuantifier.get_prediction_intervals(
            predictions, confidence_level=0.95
        )
        
        assert lower_bound.shape == (20, 5)
        assert upper_bound.shape == (20, 5)
        
        # Upper bound should be greater than lower bound
        assert torch.all(upper_bound >= lower_bound)

@pytest.mark.slow
class TestModelPerformance:
    """Performance tests for ML models"""
    
    def test_bayesian_layer_inference_speed(self):
        """Test inference speed of Bayesian layer"""
        config = {'input_dim': 512, 'output_dim': 10}
        model = BayesianDiagnosticLayer(config)
        model.eval()
        
        batch_size = 32
        input_tensor = torch.randn(batch_size, 512)
        
        import time
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(input_tensor, sample=False)
        inference_time = time.time() - start_time
        
        # Should complete 100 inferences in reasonable time
        assert inference_time < 5.0  # 5 seconds threshold
        
        # Log performance metrics
        logger.info(f"Bayesian layer inference time: {inference_time:.3f}s for 100 batches")
    
    def test_attention_memory_usage(self):
        """Test memory usage of attention mechanism"""
        config = {'fusion_dim': 512, 'attention_heads': 8}
        model = MultiModalAttention(config)
        
        batch_size = 16
        modality_features = {
            'text': torch.randn(batch_size, 768),
            'voice': torch.randn(batch_size, 1024),
            'behavioral': torch.randn(batch_size, 256)
        }
        
        # Monitor memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            output = model(modality_features)
            peak_memory = torch.cuda.memory_allocated()
            
            memory_used = peak_memory - initial_memory
            logger.info(f"Attention model memory usage: {memory_used / 1024 / 1024:.2f} MB")
            
            # Should not use excessive memory
            assert memory_used < 100 * 1024 * 1024  # 100MB threshold