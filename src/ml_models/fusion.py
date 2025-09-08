"""
Multi-modal fusion models with transformer-based attention
Moved from diagnosis/enterprise/models/fusion.py for better organization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

from .base import BaseModel, initialize_weights

logger = logging.getLogger(__name__)

class MultiModalAttention(BaseModel):
    """Transformer-based cross-modal attention mechanism for multi-modal fusion"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.d_model = config.get('fusion_dim', 512)
        self.n_heads = config.get('attention_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # Validate configuration
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        # Multi-head attention for cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Layer normalization and feed-forward network
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 4, self.d_model)
        )
        
        # Modality-specific projections with adaptive sizing
        self.modality_projections = nn.ModuleDict()
        self._setup_modality_projections()
        
        # Apply weight initialization
        self.apply(initialize_weights)
        
        logger.info(f"MultiModalAttention initialized with d_model={self.d_model}, n_heads={self.n_heads}")
    
    def _setup_modality_projections(self):
        """Setup modality-specific projection layers"""
        # Default input dimensions for each modality
        modality_dims = {
            'text': 768,      # BERT-like embeddings
            'voice': 1024,    # Voice feature dimensions
            'behavioral': 256,
            'temporal': 128,
            'physiological': 64,
            'contextual': 512
        }
        
        # Create projection layers for each modality
        for modality, input_dim in modality_dims.items():
            self.modality_projections[modality] = nn.Sequential(
                nn.Linear(input_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for multi-modal attention"""
        if not modality_features:
            raise ValueError("No modality features provided")
        
        # Project each modality to common dimension
        projected_features = []
        modality_names = []
        
        for modality, features in modality_features.items():
            if modality in self.modality_projections:
                # Ensure features have batch dimension
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                elif features.dim() > 2:
                    # Flatten extra dimensions
                    features = features.view(features.size(0), -1)
                
                try:
                    projected = self.modality_projections[modality](features)
                    projected_features.append(projected)
                    modality_names.append(modality)
                except RuntimeError as e:
                    logger.warning(f"Failed to project {modality} features: {e}")
                    continue
        
        if not projected_features:
            raise ValueError("No valid modality features could be projected")
        
        # Stack features for attention computation
        stacked_features = torch.stack(projected_features, dim=1)  # [batch, modalities, d_model]
        
        # Self-attention across modalities
        try:
            attended, attention_weights = self.cross_attention(
                stacked_features, stacked_features, stacked_features
            )
        except RuntimeError as e:
            logger.error(f"Attention computation failed: {e}")
            raise
        
        # Residual connection and layer norm
        attended = self.layer_norm1(attended + stacked_features)
        
        # Feed-forward network
        ffn_output = self.ffn(attended)
        output = self.layer_norm2(ffn_output + attended)
        
        # Global pooling across modalities with attention weighting
        # Use attention weights to create weighted average
        if attention_weights is not None:
            # Average attention weights across heads
            avg_attention = attention_weights.mean(dim=1)  # [batch, modalities, modalities]
            
            # Use self-attention weights (diagonal) for pooling weights
            self_attention = torch.diagonal(avg_attention, dim1=-2, dim2=-1)  # [batch, modalities]
            self_attention = F.softmax(self_attention, dim=-1)
            
            # Weighted pooling
            fused_representation = torch.sum(
                output * self_attention.unsqueeze(-1), dim=1
            )  # [batch, d_model]
        else:
            # Fallback to simple mean pooling
            fused_representation = torch.mean(output, dim=1)  # [batch, d_model]
        
        return fused_representation
    
    def get_attention_weights(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get attention weights for interpretability"""
        with torch.no_grad():
            # Project features
            projected_features = []
            for modality, features in modality_features.items():
                if modality in self.modality_projections:
                    if features.dim() == 1:
                        features = features.unsqueeze(0)
                    elif features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    
                    projected = self.modality_projections[modality](features)
                    projected_features.append(projected)
            
            if not projected_features:
                return None
            
            stacked_features = torch.stack(projected_features, dim=1)
            
            # Get attention weights
            _, attention_weights = self.cross_attention(
                stacked_features, stacked_features, stacked_features
            )
            
            return attention_weights

class ModalityEncoder(BaseModel):
    """Individual modality encoder with self-attention"""
    
    def __init__(self, config: Dict[str, Any], input_dim: int):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=self.dropout,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, config.get('output_dim', self.hidden_dim))
        
        self.apply(initialize_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through modality encoder"""
        # Input projection
        x = self.input_projection(x)
        
        if x.dim() == 2:
            # Add sequence dimension if not present
            x = x.unsqueeze(1)
        
        # Apply attention layers
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            attended, _ = attention(x, x, x)
            x = layer_norm(attended + x)
        
        # Remove sequence dimension if added
        if x.size(1) == 1:
            x = x.squeeze(1)
        
        # Output projection
        output = self.output_projection(x)
        
        return output

class AdaptiveFusion(BaseModel):
    """Adaptive fusion mechanism that learns optimal modality combinations"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.d_model = config.get('fusion_dim', 512)
        self.num_modalities = config.get('max_modalities', 6)
        
        # Learned modality importance weights
        self.modality_importance = nn.Parameter(torch.ones(self.num_modalities))
        
        # Fusion gate to control information flow
        # Use modality-agnostic gate operating on summed features to avoid shape mismatch
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        self.apply(initialize_weights)
    
    def forward(self, modality_features: torch.Tensor, modality_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adaptive fusion of modality features"""
        batch_size, num_modalities, d_model = modality_features.shape
        
        # Apply modality importance weights
        importance_weights = F.softmax(self.modality_importance[:num_modalities], dim=0)
        
        if modality_mask is not None:
            # Mask unavailable modalities
            importance_weights = importance_weights * modality_mask
            importance_weights = importance_weights / (importance_weights.sum() + 1e-8)
        
        # Weighted combination
        weighted_features = modality_features * importance_weights.view(1, -1, 1)

        # Sum across modalities
        summed_features = torch.sum(weighted_features, dim=1)

        # Apply fusion gate on modality-agnostic representation
        gate = self.fusion_gate(summed_features)
        
        # Apply gate
        gated_features = summed_features * gate
        
        # Final fusion
        output = self.fusion_layer(gated_features)
        
        return output