"""
Bayesian neural network components for uncertainty quantification
Moved from diagnosis/enterprise/models/bayesian.py for better organization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

from .base import BaseModel, initialize_weights

logger = logging.getLogger(__name__)

class BayesianDiagnosticLayer(BaseModel):
    """Bayesian neural network layer for uncertainty quantification in diagnostic predictions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 512)
        self.output_dim = config.get('output_dim', 5)  # Number of conditions
        self.prior_std = config.get('prior_std', 0.1)
        
        # Weight mean and log variance parameters
        self.weight_mean = nn.Parameter(torch.randn(self.output_dim, self.input_dim) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(self.output_dim, self.input_dim) * 0.1 - 2)
        
        # Bias mean and log variance parameters  
        self.bias_mean = nn.Parameter(torch.randn(self.output_dim) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(self.output_dim) * 0.1 - 2)
        
        logger.info(f"BayesianDiagnosticLayer initialized: {self.input_dim} -> {self.output_dim}")
        
    def forward(self, x: torch.Tensor, sample: bool = True, num_samples: int = 1) -> torch.Tensor:
        """Forward pass with weight sampling for uncertainty"""
        if sample and (self.training or num_samples > 1):
            outputs = []
            
            for _ in range(num_samples):
                # Sample weights from posterior distribution
                weight_std = torch.exp(0.5 * self.weight_logvar)
                weight_eps = torch.randn_like(self.weight_mean)
                weight = self.weight_mean + weight_std * weight_eps
                
                bias_std = torch.exp(0.5 * self.bias_logvar)
                bias_eps = torch.randn_like(self.bias_mean)
                bias = self.bias_mean + bias_std * bias_eps
                
                # Forward pass with sampled weights
                output = F.linear(x, weight, bias)
                outputs.append(output)
            
            if num_samples == 1:
                return outputs[0]
            else:
                return torch.stack(outputs, dim=0)  # [num_samples, batch_size, output_dim]
        else:
            # Use mean weights for deterministic output
            return F.linear(x, self.weight_mean, self.bias_mean)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior distributions"""
        # KL divergence for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mean.pow(2) + weight_var) / (self.prior_std ** 2) - 
            torch.log(weight_var / (self.prior_std ** 2)) - 1
        )
        
        # KL divergence for biases
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mean.pow(2) + bias_var) / (self.prior_std ** 2) - 
            torch.log(bias_var / (self.prior_std ** 2)) - 1
        )
        
        return weight_kl + bias_kl
    
    def get_weight_uncertainty(self) -> Dict[str, torch.Tensor]:
        """Get weight uncertainty measures"""
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        
        return {
            'weight_mean': self.weight_mean.detach(),
            'weight_std': weight_std.detach(),
            'bias_mean': self.bias_mean.detach(),
            'bias_std': bias_std.detach(),
            'weight_uncertainty': weight_std.mean().item(),
            'bias_uncertainty': bias_std.mean().item()
        }

class BayesianMLP(BaseModel):
    """Multi-layer Bayesian neural network for complex diagnostic tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 512)
        self.hidden_dims = config.get('hidden_dims', [256, 128])
        self.output_dim = config.get('output_dim', 5)
        self.prior_std = config.get('prior_std', 0.1)
        self.dropout = config.get('dropout', 0.1)
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        layer_config = {
            'input_dim': self.input_dim,
            'output_dim': self.hidden_dims[0],
            'prior_std': self.prior_std
        }
        self.layers.append(BayesianLinear(layer_config))
        
        # Hidden layers
        for i in range(1, len(self.hidden_dims)):
            layer_config = {
                'input_dim': self.hidden_dims[i-1],
                'output_dim': self.hidden_dims[i],
                'prior_std': self.prior_std
            }
            self.layers.append(BayesianLinear(layer_config))
        
        # Output layer
        layer_config = {
            'input_dim': self.hidden_dims[-1],
            'output_dim': self.output_dim,
            'prior_std': self.prior_std
        }
        self.layers.append(BayesianLinear(layer_config))
        
        # Activation and normalization
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in self.hidden_dims
        ])
        
        logger.info(f"BayesianMLP initialized with dims: {[self.input_dim] + self.hidden_dims + [self.output_dim]}")
    
    def forward(self, x: torch.Tensor, sample: bool = True, num_samples: int = 1) -> torch.Tensor:
        """Forward pass through Bayesian MLP"""
        if num_samples > 1:
            outputs = []
            for _ in range(num_samples):
                output = self._single_forward(x, sample=True)
                outputs.append(output)
            return torch.stack(outputs, dim=0)
        else:
            return self._single_forward(x, sample=sample)
    
    def _single_forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Single forward pass"""
        # Forward through hidden layers
        for i, (layer, layer_norm) in enumerate(zip(self.layers[:-1], self.layer_norms)):
            x = layer(x, sample=sample)
            x = layer_norm(x)
            x = self.activation(x)
            x = self.dropout_layer(x)
        
        # Output layer
        x = self.layers[-1](x, sample=sample)
        
        return x
    
    def total_kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence across all layers"""
        total_kl = 0
        for layer in self.layers:
            if hasattr(layer, 'kl_divergence'):
                total_kl += layer.kl_divergence()
        return total_kl

class BayesianLinear(nn.Module):
    """Single Bayesian linear layer"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.prior_std = config.get('prior_std', 0.1)
        
        # Weight parameters
        self.weight_mean = nn.Parameter(torch.randn(self.output_dim, self.input_dim) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(self.output_dim, self.input_dim) * 0.1 - 2)
        
        # Bias parameters
        self.bias_mean = nn.Parameter(torch.randn(self.output_dim) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(self.output_dim) * 0.1 - 2)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with optional weight sampling"""
        if sample and self.training:
            # Sample weights and biases
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(self.weight_mean)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mean + bias_std * torch.randn_like(self.bias_mean)
        else:
            # Use mean values
            weight = self.weight_mean
            bias = self.bias_mean
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for this layer"""
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        
        weight_kl = 0.5 * torch.sum(
            (self.weight_mean.pow(2) + weight_var) / (self.prior_std ** 2) - 
            torch.log(weight_var / (self.prior_std ** 2)) - 1
        )
        
        bias_kl = 0.5 * torch.sum(
            (self.bias_mean.pow(2) + bias_var) / (self.prior_std ** 2) - 
            torch.log(bias_var / (self.prior_std ** 2)) - 1
        )
        
        return weight_kl + bias_kl

class UncertaintyQuantifier:
    """Utility class for quantifying different types of uncertainty"""
    
    @staticmethod
    def monte_carlo_predictions(model: BayesianDiagnosticLayer, 
                              x: torch.Tensor, 
                              num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Generate Monte Carlo predictions for uncertainty estimation"""
        model.eval()
        
        with torch.no_grad():
            # Generate multiple predictions
            predictions = model(x, sample=True, num_samples=num_samples)  # [num_samples, batch, output]
            
            # Calculate statistics
            mean_pred = torch.mean(predictions, dim=0)  # [batch, output]
            std_pred = torch.std(predictions, dim=0)   # [batch, output]
            
            # Convert to probabilities
            prob_predictions = F.softmax(predictions, dim=-1)
            mean_prob = torch.mean(prob_predictions, dim=0)
            std_prob = torch.std(prob_predictions, dim=0)
            
            # Predictive entropy (epistemic uncertainty)
            log_mean_prob = torch.log(mean_prob + 1e-8)
            predictive_entropy = -torch.sum(mean_prob * log_mean_prob, dim=-1)
            
            # Mutual information (model uncertainty)
            individual_entropies = -torch.sum(
                prob_predictions * torch.log(prob_predictions + 1e-8), dim=-1
            )
            expected_entropy = torch.mean(individual_entropies, dim=0)
            mutual_info = predictive_entropy - expected_entropy
            
        return {
            'mean_logits': mean_pred,
            'std_logits': std_pred,
            'mean_probs': mean_prob,
            'std_probs': std_prob,
            'predictive_entropy': predictive_entropy,
            'mutual_information': mutual_info,
            'epistemic_uncertainty': mutual_info,
            'aleatoric_uncertainty': expected_entropy,
            'all_predictions': predictions
        }
    
    @staticmethod
    def calibration_error(predictions: torch.Tensor, 
                         targets: torch.Tensor, 
                         num_bins: int = 10) -> float:
        """Calculate calibration error for probability predictions"""
        # Get predicted probabilities and classes
        max_probs, pred_classes = torch.max(predictions, dim=1)
        
        # Calculate accuracy
        accuracies = (pred_classes == targets).float()
        
        # Bin predictions by confidence
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                # Calculate average confidence and accuracy in bin
                avg_confidence = max_probs[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                
                # Add to ECE
                ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return ece.item()
    
    @staticmethod
    def get_prediction_intervals(predictions: torch.Tensor,
                                 confidence_level: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate prediction intervals from Monte Carlo samples"""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        # Use torch.quantile for broad compatibility
        lower_bound = torch.quantile(predictions, lower_percentile / 100.0, dim=0)
        upper_bound = torch.quantile(predictions, upper_percentile / 100.0, dim=0)

        return lower_bound, upper_bound