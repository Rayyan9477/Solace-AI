"""
Advanced Pattern Recognition Engine for Therapeutic Intervention Analysis

This module implements sophisticated pattern recognition algorithms to identify
effective therapeutic approaches, user behavior patterns, and intervention sequences
using deep learning, transformer models, and graph neural networks.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import networkx as nx

from ..diagnosis.adaptive_learning import InterventionOutcome, UserProfile
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TherapeuticPattern:
    """Represents a discovered therapeutic pattern"""
    pattern_id: str
    pattern_type: str  # sequence, context, relationship, temporal
    intervention_types: List[str]
    context_conditions: Dict[str, Any]
    effectiveness_score: float
    confidence: float
    support: int  # Number of instances supporting this pattern
    discovered_at: datetime
    user_segments: List[str]  # Which user segments this pattern applies to

@dataclass
class UserBehaviorPattern:
    """Represents a user behavior pattern"""
    pattern_id: str
    user_segment: str
    behavior_indicators: List[str]
    response_patterns: Dict[str, Any]
    engagement_factors: List[str]
    optimal_interventions: List[str]
    pattern_strength: float
    temporal_aspects: Dict[str, Any]

@dataclass
class InterventionSequence:
    """Represents an effective intervention sequence"""
    sequence_id: str
    intervention_sequence: List[str]
    context_requirements: Dict[str, Any]
    success_rate: float
    average_effectiveness: float
    user_types: List[str]
    optimal_timing: Dict[str, Any]
    contraindications: List[str]

class TherapeuticTransformer(nn.Module):
    """Transformer model for analyzing therapeutic contexts and outcomes"""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, max_seq_length: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers for different tasks
        self.effectiveness_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.context_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 50)  # 50 context features
        )
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding for transformer"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer"""
        
        # Embedding and positional encoding
        embeddings = self.embedding(input_ids) * np.sqrt(self.d_model)
        seq_len = input_ids.size(1)
        embeddings += self.positional_encoding[:, :seq_len, :]
        
        # Transformer encoding
        if attention_mask is not None:
            # Convert attention mask for transformer (0 for masked positions)
            attention_mask = attention_mask == 0
        
        encoded = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        # Global representation (mean pooling over sequence)
        if attention_mask is not None:
            mask_expanded = (~attention_mask).unsqueeze(-1).expand_as(encoded).float()
            sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            global_repr = sum_embeddings / sum_mask
        else:
            global_repr = torch.mean(encoded, dim=1)
        
        # Task-specific outputs
        effectiveness_pred = self.effectiveness_predictor(global_repr)
        context_features = self.context_analyzer(global_repr)
        
        return {
            'effectiveness_prediction': effectiveness_pred,
            'context_features': context_features,
            'sequence_embeddings': encoded,
            'global_representation': global_repr
        }

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for analyzing therapeutic relationships"""
    
    def __init__(self, input_dim: int = 100, hidden_dim: int = 64, output_dim: int = 32, num_layers: int = 3):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(nn.Linear(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(nn.Linear(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GNN"""
        
        x = node_features
        
        for i in range(self.num_layers - 1):
            # Graph convolution (simplified message passing)
            x = self._graph_conv(x, edge_index, edge_weights)
            x = self.convs[i](x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self._graph_conv(x, edge_index, edge_weights)
        x = self.convs[-1](x)
        
        return x
    
    def _graph_conv(self, x: torch.Tensor, edge_index: torch.Tensor, 
                    edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simplified graph convolution operation"""
        
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        
        # Normalize edge weights
        edge_weights = deg_inv_sqrt[row] * edge_weights * deg_inv_sqrt[col]
        
        # Message passing
        out = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            out[dst] += edge_weights[i] * x[src]
        
        return out

class LSTMSequenceAnalyzer(nn.Module):
    """LSTM-based sequence analyzer for intervention patterns"""
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128, num_layers: int = 2, 
                 num_intervention_types: int = 20):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.1, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Classification heads
        self.sequence_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_intervention_types)
        )
        
        self.effectiveness_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequences: torch.Tensor, sequence_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through LSTM sequence analyzer"""
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(sequences)
        
        # Apply attention
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global representation (mean pooling over sequence)
        if sequence_lengths is not None:
            # Mask based on actual sequence lengths
            mask = torch.arange(sequences.size(1)).expand(len(sequence_lengths), sequences.size(1)) < sequence_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(attended_out).float()
            sum_embeddings = torch.sum(attended_out * mask, dim=1)
            sum_lengths = sequence_lengths.unsqueeze(-1).expand_as(sum_embeddings).float()
            global_repr = sum_embeddings / sum_lengths
        else:
            global_repr = torch.mean(attended_out, dim=1)
        
        # Task-specific outputs
        sequence_pred = self.sequence_classifier(global_repr)
        effectiveness_pred = self.effectiveness_predictor(global_repr)
        
        return {
            'sequence_prediction': sequence_pred,
            'effectiveness_prediction': effectiveness_pred,
            'sequence_embeddings': attended_out,
            'global_representation': global_repr,
            'attention_weights': attention_weights
        }

class PatternRecognitionEngine:
    """
    Advanced pattern recognition engine for therapeutic intervention analysis.
    
    Uses deep learning, graph neural networks, and traditional ML to identify
    patterns in therapeutic approaches and user behaviors.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Neural network models
        self.transformer_model = TherapeuticTransformer()
        self.graph_neural_net = GraphNeuralNetwork()
        self.sequence_analyzer = LSTMSequenceAnalyzer()
        
        # Traditional ML components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.kmeans_clusterer = KMeans(n_clusters=8, random_state=42)
        self.dbscan_clusterer = DBSCAN(eps=0.5, min_samples=5)
        
        # Pattern storage
        self.discovered_patterns = []
        self.user_behavior_patterns = []
        self.intervention_sequences = []
        
        # Knowledge graph for therapeutic relationships
        self.therapeutic_graph = nx.DiGraph()
        
        # Vocabularies and encoders
        self.intervention_vocab = {}
        self.context_vocab = {}
        self.user_segment_vocab = {}
        
        # Training state
        self.is_trained = False
        self.training_data_cache = []
        
        # Load pre-trained models if available
        self._load_pretrained_models()
    
    async def analyze_therapeutic_patterns(self,
                                         user_interactions: List[Dict],
                                         outcomes: List[InterventionOutcome]) -> Dict[str, Any]:
        """
        Main method for analyzing therapeutic patterns and their effectiveness.
        """
        try:
            self.logger.info(f"Analyzing therapeutic patterns from {len(user_interactions)} interactions")
            
            # Prepare data for analysis
            prepared_data = await self._prepare_data(user_interactions, outcomes)
            
            # Sequential pattern analysis
            sequence_patterns = await self._analyze_intervention_sequences(prepared_data)
            
            # Contextual pattern analysis
            context_patterns = await self._analyze_contextual_patterns(prepared_data)
            
            # Relationship analysis using graph neural network
            relationship_patterns = await self._analyze_therapeutic_relationships(prepared_data)
            
            # Temporal pattern analysis
            temporal_patterns = await self._analyze_temporal_patterns(prepared_data)
            
            # User behavior clustering
            user_behavior_clusters = await self._analyze_user_behavior_patterns(prepared_data)
            
            # Generate optimal intervention sequences
            optimal_sequences = await self._generate_optimal_sequences(prepared_data)
            
            # Calculate pattern confidence and support
            pattern_statistics = self._calculate_pattern_statistics(
                sequence_patterns, context_patterns, relationship_patterns
            )
            
            return {
                'sequence_patterns': sequence_patterns,
                'context_patterns': context_patterns,
                'relationship_patterns': relationship_patterns,
                'temporal_patterns': temporal_patterns,
                'user_behavior_clusters': user_behavior_clusters,
                'optimal_sequences': optimal_sequences,
                'pattern_statistics': pattern_statistics,
                'total_patterns_discovered': len(self.discovered_patterns),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in therapeutic pattern analysis: {str(e)}")
            return {'error': str(e)}
    
    async def _prepare_data(self, 
                          user_interactions: List[Dict],
                          outcomes: List[InterventionOutcome]) -> Dict[str, Any]:
        """Prepare and encode data for pattern analysis"""
        
        # Build vocabularies
        self._build_vocabularies(user_interactions, outcomes)
        
        # Encode interventions and contexts
        encoded_interactions = []
        for i, interaction in enumerate(user_interactions):
            if i < len(outcomes):
                outcome = outcomes[i]
                encoded_interaction = {
                    'intervention_type': self.intervention_vocab.get(interaction['intervention_type'], 0),
                    'context_vector': self._encode_context(interaction['context']),
                    'user_segment': self._determine_user_segment(interaction.get('user_id')),
                    'effectiveness_score': outcome.effectiveness_score,
                    'engagement_score': outcome.engagement_score,
                    'timestamp': interaction.get('timestamp', datetime.now()),
                    'breakthrough_indicator': outcome.breakthrough_indicator,
                    'user_response_vector': self._encode_user_response(outcome.user_response)
                }
                encoded_interactions.append(encoded_interaction)
        
        # Create feature matrices
        feature_matrix = self._create_feature_matrix(encoded_interactions)
        
        # Create temporal sequences
        temporal_sequences = self._create_temporal_sequences(encoded_interactions)
        
        # Create graph data
        graph_data = self._create_graph_representation(encoded_interactions)
        
        return {
            'encoded_interactions': encoded_interactions,
            'feature_matrix': feature_matrix,
            'temporal_sequences': temporal_sequences,
            'graph_data': graph_data,
            'raw_interactions': user_interactions,
            'raw_outcomes': outcomes
        }
    
    def _build_vocabularies(self, 
                           user_interactions: List[Dict],
                           outcomes: List[InterventionOutcome]) -> None:
        """Build vocabularies for encoding"""
        
        # Intervention types vocabulary
        intervention_types = set()
        for interaction in user_interactions:
            intervention_types.add(interaction.get('intervention_type', 'unknown'))
        
        for outcome in outcomes:
            intervention_types.add(outcome.intervention_type)
        
        self.intervention_vocab = {itype: i for i, itype in enumerate(sorted(intervention_types))}
        
        # Context vocabulary (simplified)
        context_keys = set()
        for interaction in user_interactions:
            if 'context' in interaction and isinstance(interaction['context'], dict):
                context_keys.update(interaction['context'].keys())
        
        for outcome in outcomes:
            if isinstance(outcome.context, dict):
                context_keys.update(outcome.context.keys())
        
        self.context_vocab = {key: i for i, key in enumerate(sorted(context_keys))}
        
        self.logger.debug(f"Built vocabularies: {len(self.intervention_vocab)} interventions, {len(self.context_vocab)} context keys")
    
    def _encode_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode context dictionary into feature vector"""
        
        vector = np.zeros(max(50, len(self.context_vocab)))  # Fixed size vector
        
        if isinstance(context, dict):
            for key, value in context.items():
                key_idx = self.context_vocab.get(key)
                if key_idx is not None and key_idx < len(vector):
                    if isinstance(value, (int, float)):
                        vector[key_idx] = float(value)
                    elif isinstance(value, bool):
                        vector[key_idx] = float(value)
                    elif isinstance(value, str):
                        # Simple string encoding (could be improved with embeddings)
                        vector[key_idx] = hash(value) % 100 / 100.0
        
        return vector
    
    def _encode_user_response(self, user_response: str) -> np.ndarray:
        """Encode user response into feature vector"""
        
        if not user_response:
            return np.zeros(20)
        
        # Simple response encoding (could be improved with NLP models)
        features = np.zeros(20)
        
        response_lower = user_response.lower()
        
        # Sentiment indicators
        positive_words = ['good', 'great', 'helpful', 'better', 'thanks', 'positive']
        negative_words = ['bad', 'worse', 'difficult', 'hard', 'negative', 'frustrated']
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        features[0] = positive_count / max(1, len(response_lower.split()))
        features[1] = negative_count / max(1, len(response_lower.split()))
        features[2] = len(user_response.split())  # Response length
        features[3] = len(user_response)  # Character count
        
        # Engagement indicators
        engagement_words = ['understand', 'realize', 'see', 'makes sense', 'clear']
        engagement_count = sum(1 for word in engagement_words if word in response_lower)
        features[4] = engagement_count / max(1, len(response_lower.split()))
        
        return features
    
    def _determine_user_segment(self, user_id: str) -> str:
        """Determine user segment (simplified clustering)"""
        # In a real implementation, this would use user profiling
        # For now, return a simple segmentation
        if user_id:
            segment_idx = hash(user_id) % 5
            return f"segment_{segment_idx}"
        return "unknown"
    
    def _create_feature_matrix(self, encoded_interactions: List[Dict]) -> np.ndarray:
        """Create feature matrix for traditional ML algorithms"""
        
        if not encoded_interactions:
            return np.array([]).reshape(0, 0)
        
        features = []
        for interaction in encoded_interactions:
            feature_vector = np.concatenate([
                [interaction['intervention_type']],
                interaction['context_vector'],
                interaction['user_response_vector'],
                [interaction['effectiveness_score']],
                [interaction['engagement_score']],
                [float(interaction['breakthrough_indicator'])],
                [hash(interaction['user_segment']) % 100]  # Simple segment encoding
            ])
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_temporal_sequences(self, encoded_interactions: List[Dict]) -> List[List[Dict]]:
        """Create temporal sequences grouped by user"""
        
        user_sequences = defaultdict(list)
        
        for interaction in encoded_interactions:
            user_id = interaction.get('user_id', 'unknown')
            user_sequences[user_id].append(interaction)
        
        # Sort by timestamp
        for user_id in user_sequences:
            user_sequences[user_id].sort(key=lambda x: x.get('timestamp', datetime.now()))
        
        return list(user_sequences.values())
    
    def _create_graph_representation(self, encoded_interactions: List[Dict]) -> Dict[str, Any]:
        """Create graph representation of therapeutic relationships"""
        
        # Build intervention co-occurrence graph
        intervention_graph = nx.Graph()
        
        # Add nodes (intervention types)
        for interaction in encoded_interactions:
            intervention_type = interaction['intervention_type']
            if not intervention_graph.has_node(intervention_type):
                intervention_graph.add_node(intervention_type, 
                                          effectiveness_scores=[],
                                          engagement_scores=[])
            
            # Add effectiveness and engagement scores to node
            intervention_graph.nodes[intervention_type]['effectiveness_scores'].append(
                interaction['effectiveness_score']
            )
            intervention_graph.nodes[intervention_type]['engagement_scores'].append(
                interaction['engagement_score']
            )
        
        # Add edges based on sequential relationships
        sequences = self._create_temporal_sequences(encoded_interactions)
        
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current_intervention = sequence[i]['intervention_type']
                next_intervention = sequence[i + 1]['intervention_type']
                
                if intervention_graph.has_edge(current_intervention, next_intervention):
                    intervention_graph[current_intervention][next_intervention]['weight'] += 1
                else:
                    intervention_graph.add_edge(current_intervention, next_intervention, weight=1)
        
        # Convert to format suitable for GNN
        node_features = []
        node_mapping = {}
        
        for i, (node, data) in enumerate(intervention_graph.nodes(data=True)):
            node_mapping[node] = i
            
            # Node features: [avg_effectiveness, avg_engagement, num_occurrences]
            effectiveness_scores = data.get('effectiveness_scores', [])
            engagement_scores = data.get('engagement_scores', [])
            
            features = [
                np.mean(effectiveness_scores) if effectiveness_scores else 0.0,
                np.mean(engagement_scores) if engagement_scores else 0.0,
                len(effectiveness_scores),
                np.std(effectiveness_scores) if len(effectiveness_scores) > 1 else 0.0,
                np.std(engagement_scores) if len(engagement_scores) > 1 else 0.0
            ]
            
            # Pad to fixed size
            features += [0.0] * (100 - len(features))  # Pad to 100 features
            node_features.append(features[:100])
        
        # Create edge index and weights
        edge_index = []
        edge_weights = []
        
        for edge in intervention_graph.edges(data=True):
            src_idx = node_mapping[edge[0]]
            dst_idx = node_mapping[edge[1]]
            weight = edge[2]['weight']
            
            edge_index.extend([[src_idx, dst_idx], [dst_idx, src_idx]])  # Undirected
            edge_weights.extend([weight, weight])
        
        return {
            'node_features': np.array(node_features),
            'edge_index': np.array(edge_index).T if edge_index else np.array([]).reshape(2, 0),
            'edge_weights': np.array(edge_weights),
            'node_mapping': node_mapping,
            'graph': intervention_graph
        }
    
    async def _analyze_intervention_sequences(self, prepared_data: Dict[str, Any]) -> List[InterventionSequence]:
        """Analyze intervention sequences using LSTM"""
        
        sequences = prepared_data['temporal_sequences']
        discovered_sequences = []
        
        # Find frequent sequences using simple pattern mining
        sequence_patterns = defaultdict(list)
        
        for sequence in sequences:
            if len(sequence) >= 2:
                for i in range(len(sequence) - 1):
                    seq_pair = (sequence[i]['intervention_type'], sequence[i + 1]['intervention_type'])
                    sequence_patterns[seq_pair].append({
                        'effectiveness_improvement': sequence[i + 1]['effectiveness_score'] - sequence[i]['effectiveness_score'],
                        'context': sequence[i]['context_vector'],
                        'user_segment': sequence[i]['user_segment']
                    })
        
        # Analyze patterns
        for seq_pattern, instances in sequence_patterns.items():
            if len(instances) >= 3:  # Minimum support threshold
                
                effectiveness_improvements = [inst['effectiveness_improvement'] for inst in instances]
                avg_improvement = np.mean(effectiveness_improvements)
                
                if avg_improvement > 0.1:  # Significant improvement threshold
                    
                    # Determine user types and contexts
                    user_segments = list(set(inst['user_segment'] for inst in instances))
                    
                    # Calculate success rate
                    successful_instances = sum(1 for imp in effectiveness_improvements if imp > 0)
                    success_rate = successful_instances / len(instances)
                    
                    intervention_sequence = InterventionSequence(
                        sequence_id=f"seq_{seq_pattern[0]}_{seq_pattern[1]}_{int(datetime.now().timestamp())}",
                        intervention_sequence=[
                            next(k for k, v in self.intervention_vocab.items() if v == seq_pattern[0]),
                            next(k for k, v in self.intervention_vocab.items() if v == seq_pattern[1])
                        ],
                        context_requirements={
                            'min_baseline_effectiveness': 0.3,
                            'applicable_segments': user_segments
                        },
                        success_rate=success_rate,
                        average_effectiveness=avg_improvement,
                        user_types=user_segments,
                        optimal_timing={'min_interval_hours': 24, 'max_interval_hours': 168},
                        contraindications=[]
                    )
                    
                    discovered_sequences.append(intervention_sequence)
        
        self.intervention_sequences.extend(discovered_sequences)
        return discovered_sequences
    
    async def _analyze_contextual_patterns(self, prepared_data: Dict[str, Any]) -> List[TherapeuticPattern]:
        """Analyze contextual patterns using clustering and statistical analysis"""
        
        feature_matrix = prepared_data['feature_matrix']
        encoded_interactions = prepared_data['encoded_interactions']
        
        if len(feature_matrix) == 0:
            return []
        
        # Standardize features
        try:
            standardized_features = self.scaler.fit_transform(feature_matrix)
        except ValueError:
            self.logger.warning("Could not standardize features, using raw features")
            standardized_features = feature_matrix
        
        # Perform clustering to find context patterns
        try:
            cluster_labels = self.kmeans_clusterer.fit_predict(standardized_features)
        except Exception as e:
            self.logger.warning(f"K-means clustering failed: {str(e)}")
            cluster_labels = np.zeros(len(feature_matrix))
        
        # Analyze each cluster
        contextual_patterns = []
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_interactions = [encoded_interactions[i] for i in np.where(cluster_mask)[0]]
            
            if len(cluster_interactions) >= 3:  # Minimum cluster size
                
                # Calculate cluster statistics
                effectiveness_scores = [int['effectiveness_score'] for int in cluster_interactions]
                engagement_scores = [int['engagement_score'] for int in cluster_interactions]
                
                avg_effectiveness = np.mean(effectiveness_scores)
                avg_engagement = np.mean(engagement_scores)
                
                if avg_effectiveness > 0.6:  # Only keep effective patterns
                    
                    # Identify common intervention types in this cluster
                    intervention_types = [int['intervention_type'] for int in cluster_interactions]
                    intervention_counts = defaultdict(int)
                    for itype in intervention_types:
                        intervention_counts[itype] += 1
                    
                    # Get most common interventions
                    common_interventions = sorted(intervention_counts.items(), 
                                                key=lambda x: x[1], reverse=True)[:3]
                    
                    intervention_names = []
                    for int_code, count in common_interventions:
                        int_name = next((k for k, v in self.intervention_vocab.items() if v == int_code), 'unknown')
                        intervention_names.append(int_name)
                    
                    # Identify context conditions (simplified)
                    context_conditions = {
                        'avg_effectiveness_threshold': avg_effectiveness - 0.1,
                        'avg_engagement_threshold': avg_engagement - 0.1,
                        'cluster_size': len(cluster_interactions)
                    }
                    
                    # User segments in this cluster
                    user_segments = list(set(int['user_segment'] for int in cluster_interactions))
                    
                    pattern = TherapeuticPattern(
                        pattern_id=f"context_pattern_{cluster_id}_{int(datetime.now().timestamp())}",
                        pattern_type="context",
                        intervention_types=intervention_names,
                        context_conditions=context_conditions,
                        effectiveness_score=avg_effectiveness,
                        confidence=min(1.0, len(cluster_interactions) / 10.0),  # Confidence based on support
                        support=len(cluster_interactions),
                        discovered_at=datetime.now(),
                        user_segments=user_segments
                    )
                    
                    contextual_patterns.append(pattern)
        
        self.discovered_patterns.extend(contextual_patterns)
        return contextual_patterns
    
    async def _analyze_therapeutic_relationships(self, prepared_data: Dict[str, Any]) -> List[TherapeuticPattern]:
        """Analyze therapeutic relationships using graph neural network"""
        
        graph_data = prepared_data['graph_data']
        
        if len(graph_data['node_features']) == 0:
            return []
        
        # Convert to tensors
        node_features = torch.FloatTensor(graph_data['node_features'])
        edge_index = torch.LongTensor(graph_data['edge_index'])
        edge_weights = torch.FloatTensor(graph_data['edge_weights'])
        
        # Forward pass through GNN
        try:
            with torch.no_grad():
                node_embeddings = self.graph_neural_net(node_features, edge_index, edge_weights)
            
            # Analyze embeddings to find relationship patterns
            embeddings_np = node_embeddings.numpy()
            
            # Cluster nodes based on embeddings
            if len(embeddings_np) > 1:
                try:
                    cluster_labels = self.dbscan_clusterer.fit_predict(embeddings_np)
                except Exception:
                    cluster_labels = np.zeros(len(embeddings_np))
            else:
                cluster_labels = np.zeros(len(embeddings_np))
            
            relationship_patterns = []
            
            # Analyze each cluster
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:  # Skip noise points in DBSCAN
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_nodes = np.where(cluster_mask)[0]
                
                if len(cluster_nodes) >= 2:
                    # Get intervention names for nodes in cluster
                    node_mapping_reverse = {v: k for k, v in graph_data['node_mapping'].items()}
                    intervention_names = []
                    
                    for node_idx in cluster_nodes:
                        int_code = node_mapping_reverse.get(node_idx)
                        if int_code is not None:
                            int_name = next((k for k, v in self.intervention_vocab.items() if v == int_code), 'unknown')
                            intervention_names.append(int_name)
                    
                    # Calculate cluster effectiveness
                    cluster_effectiveness = np.mean([node_features[i][0].item() for i in cluster_nodes])
                    
                    if cluster_effectiveness > 0.5:  # Only keep effective clusters
                        
                        pattern = TherapeuticPattern(
                            pattern_id=f"relationship_pattern_{cluster_id}_{int(datetime.now().timestamp())}",
                            pattern_type="relationship",
                            intervention_types=intervention_names,
                            context_conditions={
                                'cluster_effectiveness': cluster_effectiveness,
                                'relationship_strength': 'high'
                            },
                            effectiveness_score=cluster_effectiveness,
                            confidence=min(1.0, len(cluster_nodes) / 5.0),
                            support=len(cluster_nodes),
                            discovered_at=datetime.now(),
                            user_segments=["general"]  # Simplified for relationships
                        )
                        
                        relationship_patterns.append(pattern)
            
            return relationship_patterns
            
        except Exception as e:
            self.logger.error(f"Error in GNN analysis: {str(e)}")
            return []
    
    async def _analyze_temporal_patterns(self, prepared_data: Dict[str, Any]) -> List[TherapeuticPattern]:
        """Analyze temporal patterns in interventions"""
        
        sequences = prepared_data['temporal_sequences']
        temporal_patterns = []
        
        # Analyze time-based patterns
        time_intervals = []
        effectiveness_changes = []
        
        for sequence in sequences:
            if len(sequence) >= 2:
                for i in range(len(sequence) - 1):
                    # Calculate time interval
                    time_diff = (sequence[i + 1]['timestamp'] - sequence[i]['timestamp']).total_seconds() / 3600  # Hours
                    effectiveness_change = sequence[i + 1]['effectiveness_score'] - sequence[i]['effectiveness_score']
                    
                    time_intervals.append(time_diff)
                    effectiveness_changes.append(effectiveness_change)
        
        if time_intervals:
            # Find optimal time intervals
            positive_changes = [(interval, change) for interval, change in zip(time_intervals, effectiveness_changes) if change > 0]
            
            if positive_changes:
                # Group by time intervals
                interval_groups = defaultdict(list)
                
                for interval, change in positive_changes:
                    # Bin intervals
                    if interval < 6:
                        interval_groups['immediate'].append(change)
                    elif interval < 24:
                        interval_groups['same_day'].append(change)
                    elif interval < 168:
                        interval_groups['same_week'].append(change)
                    else:
                        interval_groups['long_term'].append(change)
                
                # Analyze each group
                for interval_type, changes in interval_groups.items():
                    if len(changes) >= 3:  # Minimum support
                        avg_change = np.mean(changes)
                        
                        if avg_change > 0.1:  # Significant improvement
                            pattern = TherapeuticPattern(
                                pattern_id=f"temporal_pattern_{interval_type}_{int(datetime.now().timestamp())}",
                                pattern_type="temporal",
                                intervention_types=["sequential"],
                                context_conditions={
                                    'optimal_interval': interval_type,
                                    'avg_improvement': avg_change
                                },
                                effectiveness_score=avg_change,
                                confidence=min(1.0, len(changes) / 10.0),
                                support=len(changes),
                                discovered_at=datetime.now(),
                                user_segments=["general"]
                            )
                            
                            temporal_patterns.append(pattern)
        
        return temporal_patterns
    
    async def _analyze_user_behavior_patterns(self, prepared_data: Dict[str, Any]) -> List[UserBehaviorPattern]:
        """Analyze user behavior patterns and segment users"""
        
        encoded_interactions = prepared_data['encoded_interactions']
        
        # Group interactions by user segment
        segment_data = defaultdict(list)
        
        for interaction in encoded_interactions:
            segment = interaction['user_segment']
            segment_data[segment].append(interaction)
        
        behavior_patterns = []
        
        # Analyze each user segment
        for segment, interactions in segment_data.items():
            if len(interactions) >= 5:  # Minimum data per segment
                
                # Calculate segment statistics
                effectiveness_scores = [int['effectiveness_score'] for int in interactions]
                engagement_scores = [int['engagement_score'] for int in interactions]
                breakthrough_count = sum(1 for int in interactions if int['breakthrough_indicator'])
                
                avg_effectiveness = np.mean(effectiveness_scores)
                avg_engagement = np.mean(engagement_scores)
                breakthrough_rate = breakthrough_count / len(interactions)
                
                # Identify preferred interventions
                intervention_preferences = defaultdict(list)
                for interaction in interactions:
                    intervention_preferences[interaction['intervention_type']].append(
                        interaction['effectiveness_score']
                    )
                
                # Find optimal interventions for this segment
                optimal_interventions = []
                for int_type, scores in intervention_preferences.items():
                    if len(scores) >= 2 and np.mean(scores) > 0.6:
                        int_name = next((k for k, v in self.intervention_vocab.items() if v == int_type), 'unknown')
                        optimal_interventions.append(int_name)
                
                # Identify engagement factors (simplified)
                engagement_factors = []
                if avg_engagement > 0.7:
                    engagement_factors.append("high_baseline_engagement")
                if breakthrough_rate > 0.15:
                    engagement_factors.append("breakthrough_responsive")
                if np.std(effectiveness_scores) < 0.2:
                    engagement_factors.append("consistent_response")
                
                behavior_pattern = UserBehaviorPattern(
                    pattern_id=f"behavior_{segment}_{int(datetime.now().timestamp())}",
                    user_segment=segment,
                    behavior_indicators=[
                        f"avg_effectiveness_{avg_effectiveness:.2f}",
                        f"avg_engagement_{avg_engagement:.2f}",
                        f"breakthrough_rate_{breakthrough_rate:.2f}"
                    ],
                    response_patterns={
                        'effectiveness_mean': avg_effectiveness,
                        'effectiveness_std': np.std(effectiveness_scores),
                        'engagement_mean': avg_engagement,
                        'engagement_std': np.std(engagement_scores)
                    },
                    engagement_factors=engagement_factors,
                    optimal_interventions=optimal_interventions,
                    pattern_strength=min(1.0, len(interactions) / 20.0),
                    temporal_aspects={
                        'interaction_count': len(interactions),
                        'time_span_days': (max(int['timestamp'] for int in interactions) - 
                                         min(int['timestamp'] for int in interactions)).days
                    }
                )
                
                behavior_patterns.append(behavior_pattern)
        
        self.user_behavior_patterns.extend(behavior_patterns)
        return behavior_patterns
    
    async def _generate_optimal_sequences(self, prepared_data: Dict[str, Any]) -> List[InterventionSequence]:
        """Generate optimal intervention sequences based on discovered patterns"""
        
        # Use discovered patterns to generate recommendations
        optimal_sequences = []
        
        # Combine insights from all pattern types
        sequence_patterns = [p for p in self.discovered_patterns if p.pattern_type == "sequence"]
        context_patterns = [p for p in self.discovered_patterns if p.pattern_type == "context"]
        
        # Generate sequences by combining high-effectiveness patterns
        for seq_pattern in sequence_patterns:
            for context_pattern in context_patterns:
                # Check if patterns are compatible
                if (seq_pattern.effectiveness_score > 0.7 and 
                    context_pattern.effectiveness_score > 0.7):
                    
                    # Create combined sequence
                    combined_sequence = InterventionSequence(
                        sequence_id=f"optimal_{seq_pattern.pattern_id}_{context_pattern.pattern_id}",
                        intervention_sequence=seq_pattern.intervention_types + context_pattern.intervention_types,
                        context_requirements={
                            **seq_pattern.context_conditions,
                            **context_pattern.context_conditions
                        },
                        success_rate=min(seq_pattern.effectiveness_score, context_pattern.effectiveness_score),
                        average_effectiveness=(seq_pattern.effectiveness_score + context_pattern.effectiveness_score) / 2,
                        user_types=list(set(seq_pattern.user_segments) & set(context_pattern.user_segments)),
                        optimal_timing={'min_interval_hours': 12, 'max_interval_hours': 72},
                        contraindications=[]
                    )
                    
                    optimal_sequences.append(combined_sequence)
        
        return optimal_sequences[:10]  # Limit to top 10 sequences
    
    def _calculate_pattern_statistics(self, 
                                    sequence_patterns: List[InterventionSequence],
                                    context_patterns: List[TherapeuticPattern],
                                    relationship_patterns: List[TherapeuticPattern]) -> Dict[str, Any]:
        """Calculate statistics about discovered patterns"""
        
        all_patterns = context_patterns + relationship_patterns
        
        if not all_patterns:
            return {"message": "No patterns discovered"}
        
        effectiveness_scores = [p.effectiveness_score for p in all_patterns]
        confidence_scores = [p.confidence for p in all_patterns]
        support_counts = [p.support for p in all_patterns]
        
        statistics = {
            'total_patterns': len(all_patterns),
            'sequence_patterns': len(sequence_patterns),
            'context_patterns': len([p for p in all_patterns if p.pattern_type == "context"]),
            'relationship_patterns': len([p for p in all_patterns if p.pattern_type == "relationship"]),
            'effectiveness_statistics': {
                'mean': np.mean(effectiveness_scores),
                'std': np.std(effectiveness_scores),
                'min': np.min(effectiveness_scores),
                'max': np.max(effectiveness_scores)
            },
            'confidence_statistics': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            },
            'support_statistics': {
                'mean': np.mean(support_counts),
                'std': np.std(support_counts),
                'min': np.min(support_counts),
                'max': np.max(support_counts)
            },
            'high_quality_patterns': len([p for p in all_patterns if p.effectiveness_score > 0.7 and p.confidence > 0.6])
        }
        
        return statistics
    
    def _load_pretrained_models(self) -> None:
        """Load pre-trained models if available"""
        
        model_dir = self.config.get('model_directory', 'src/data/adaptive_learning/pattern_models')
        
        # Try to load transformer model
        transformer_path = f"{model_dir}/transformer_model.pt"
        if os.path.exists(transformer_path):
            try:
                self.transformer_model.load_state_dict(torch.load(transformer_path))
                self.logger.info("Loaded pre-trained transformer model")
            except Exception as e:
                self.logger.warning(f"Could not load transformer model: {str(e)}")
        
        # Try to load GNN model
        gnn_path = f"{model_dir}/gnn_model.pt"
        if os.path.exists(gnn_path):
            try:
                self.graph_neural_net.load_state_dict(torch.load(gnn_path))
                self.logger.info("Loaded pre-trained GNN model")
            except Exception as e:
                self.logger.warning(f"Could not load GNN model: {str(e)}")
        
        # Try to load LSTM model
        lstm_path = f"{model_dir}/lstm_model.pt"
        if os.path.exists(lstm_path):
            try:
                self.sequence_analyzer.load_state_dict(torch.load(lstm_path))
                self.logger.info("Loaded pre-trained LSTM model")
            except Exception as e:
                self.logger.warning(f"Could not load LSTM model: {str(e)}")
    
    def save_models(self, model_dir: str = None) -> None:
        """Save trained models to disk"""
        
        if model_dir is None:
            model_dir = self.config.get('model_directory', 'src/data/adaptive_learning/pattern_models')
        
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Save transformer model
            torch.save(self.transformer_model.state_dict(), f"{model_dir}/transformer_model.pt")
            
            # Save GNN model
            torch.save(self.graph_neural_net.state_dict(), f"{model_dir}/gnn_model.pt")
            
            # Save LSTM model
            torch.save(self.sequence_analyzer.state_dict(), f"{model_dir}/lstm_model.pt")
            
            # Save discovered patterns
            with open(f"{model_dir}/discovered_patterns.pkl", "wb") as f:
                pickle.dump({
                    'therapeutic_patterns': [asdict(p) for p in self.discovered_patterns],
                    'behavior_patterns': [asdict(p) for p in self.user_behavior_patterns],
                    'intervention_sequences': [asdict(p) for p in self.intervention_sequences],
                    'vocabularies': {
                        'intervention_vocab': self.intervention_vocab,
                        'context_vocab': self.context_vocab,
                        'user_segment_vocab': self.user_segment_vocab
                    }
                }, f)
            
            self.logger.info(f"Models saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of discovered patterns"""
        
        return {
            'total_therapeutic_patterns': len(self.discovered_patterns),
            'total_behavior_patterns': len(self.user_behavior_patterns),
            'total_intervention_sequences': len(self.intervention_sequences),
            'pattern_types': {
                'sequence': len([p for p in self.discovered_patterns if p.pattern_type == "sequence"]),
                'context': len([p for p in self.discovered_patterns if p.pattern_type == "context"]),
                'relationship': len([p for p in self.discovered_patterns if p.pattern_type == "relationship"]),
                'temporal': len([p for p in self.discovered_patterns if p.pattern_type == "temporal"])
            },
            'high_confidence_patterns': len([p for p in self.discovered_patterns if p.confidence > 0.8]),
            'high_effectiveness_patterns': len([p for p in self.discovered_patterns if p.effectiveness_score > 0.8]),
            'vocabularies_built': bool(self.intervention_vocab and self.context_vocab),
            'models_trained': self.is_trained
        }