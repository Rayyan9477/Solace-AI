"""
Advanced Personalization Engine for User-Specific Response Optimization

This module implements sophisticated personalization algorithms to adapt therapeutic
interventions, agent responses, and system behavior based on individual user patterns,
preferences, and outcomes. It uses collaborative filtering, deep learning, and
reinforcement learning to create highly personalized experiences.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import pickle
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..diagnosis.adaptive_learning import InterventionOutcome, UserProfile
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class PersonalizationProfile:
    """Comprehensive personalization profile for a user"""
    user_id: str
    demographic_features: Dict[str, Any]
    behavioral_features: Dict[str, Any]
    preference_vector: np.ndarray
    response_patterns: Dict[str, Any]
    intervention_effectiveness: Dict[str, float]
    learning_style: str
    engagement_preferences: Dict[str, Any]
    temporal_preferences: Dict[str, Any]
    cultural_context: Dict[str, Any]
    personality_traits: Dict[str, float]
    therapeutic_goals: List[str]
    contraindications: List[str]
    adaptation_rate: float
    confidence_scores: Dict[str, float]
    last_updated: datetime
    profile_version: int

@dataclass
class PersonalizationRecommendation:
    """Personalization recommendation for a specific context"""
    recommendation_id: str
    user_id: str
    recommended_interventions: List[str]
    intervention_weights: Dict[str, float]
    personalization_factors: Dict[str, Any]
    confidence: float
    expected_effectiveness: float
    reasoning: str
    contextual_adaptations: Dict[str, Any]
    timing_recommendations: Dict[str, Any]
    fallback_options: List[str]
    generated_at: datetime
    expires_at: datetime

@dataclass
class UserCluster:
    """User cluster for collaborative personalization"""
    cluster_id: str
    cluster_name: str
    member_user_ids: Set[str]
    cluster_characteristics: Dict[str, Any]
    common_preferences: Dict[str, float]
    effective_interventions: List[str]
    cluster_size: int
    coherence_score: float
    representative_features: np.ndarray
    last_updated: datetime

class DeepPersonalizationNetwork(nn.Module):
    """Deep neural network for learning user personalization patterns"""
    
    def __init__(self, 
                 user_feature_dim: int = 100,
                 intervention_dim: int = 50,
                 context_dim: int = 75,
                 hidden_dims: List[int] = [256, 128, 64],
                 num_intervention_types: int = 20):
        super().__init__()
        
        self.user_feature_dim = user_feature_dim
        self.intervention_dim = intervention_dim
        self.context_dim = context_dim
        
        # User embedding layer
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 64)
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], 32)
        )
        
        # Intervention encoder
        self.intervention_encoder = nn.Embedding(num_intervention_types, intervention_dim)
        
        # Fusion network
        fusion_input_dim = 64 + 32 + intervention_dim  # user + context + intervention
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 32)
        )
        
        # Output heads
        self.effectiveness_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.preference_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        
        self.engagement_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for personalization factors
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
    
    def forward(self, user_features: torch.Tensor, context_features: torch.Tensor,
                intervention_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through personalization network"""
        
        # Encode inputs
        user_encoded = self.user_encoder(user_features)
        context_encoded = self.context_encoder(context_features)
        intervention_encoded = self.intervention_encoder(intervention_ids)
        
        # Fuse representations
        fused_input = torch.cat([user_encoded, context_encoded, intervention_encoded], dim=-1)
        fused_representation = self.fusion_network(fused_input)
        
        # Apply self-attention for personalization factor weighting
        attended_repr, attention_weights = self.attention(
            fused_representation.unsqueeze(1),
            fused_representation.unsqueeze(1),
            fused_representation.unsqueeze(1)
        )
        attended_repr = attended_repr.squeeze(1)
        
        # Generate predictions
        effectiveness_pred = self.effectiveness_predictor(attended_repr)
        preference_pred = self.preference_predictor(attended_repr)
        engagement_pred = self.engagement_predictor(attended_repr)
        
        return {
            'effectiveness_prediction': effectiveness_pred,
            'preference_prediction': preference_pred,
            'engagement_prediction': engagement_pred,
            'user_encoding': user_encoded,
            'context_encoding': context_encoded,
            'intervention_encoding': intervention_encoded,
            'fused_representation': fused_representation,
            'attention_weights': attention_weights
        }

class CollaborativeFilteringEngine:
    """Collaborative filtering for user similarity and recommendation"""
    
    def __init__(self, n_neighbors: int = 10, similarity_threshold: float = 0.3):
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.user_index = {}
        self.item_index = {}
        self.fitted = False
        self.logger = get_logger(__name__)
    
    def fit(self, user_intervention_ratings: Dict[str, Dict[str, float]]) -> None:
        """Fit collaborative filtering model on user-intervention ratings"""
        
        # Build user and item indices
        users = sorted(user_intervention_ratings.keys())
        all_interventions = set()
        
        for user_ratings in user_intervention_ratings.values():
            all_interventions.update(user_ratings.keys())
        
        interventions = sorted(all_interventions)
        
        self.user_index = {user: i for i, user in enumerate(users)}
        self.item_index = {item: i for i, item in enumerate(interventions)}
        
        # Create user-item matrix
        n_users, n_items = len(users), len(interventions)
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for user, ratings in user_intervention_ratings.items():
            user_idx = self.user_index[user]
            for intervention, rating in ratings.items():
                item_idx = self.item_index[intervention]
                self.user_item_matrix[user_idx, item_idx] = rating
        
        # Compute user similarity matrix
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        
        # Compute item similarity matrix
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        # Fit KNN model for efficient neighbor finding
        self.knn_model.fit(self.user_item_matrix)
        
        self.fitted = True
        self.logger.info(f"Collaborative filtering model fitted with {n_users} users and {n_items} interventions")
    
    def get_similar_users(self, user_id: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        """Get similar users to the given user"""
        
        if not self.fitted or user_id not in self.user_index:
            return []
        
        user_idx = self.user_index[user_id]
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Get indices of most similar users (excluding self)
        similar_indices = np.argsort(user_similarities)[::-1][1:n_similar + 1]
        
        # Convert back to user IDs with similarity scores
        reverse_user_index = {v: k for k, v in self.user_index.items()}
        similar_users = [
            (reverse_user_index[idx], user_similarities[idx])
            for idx in similar_indices
            if user_similarities[idx] > self.similarity_threshold
        ]
        
        return similar_users
    
    def recommend_interventions(self, user_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """Recommend interventions for a user using collaborative filtering"""
        
        if not self.fitted or user_id not in self.user_index:
            return []
        
        user_idx = self.user_index[user_id]
        user_ratings = self.user_item_matrix[user_idx]
        
        # Find similar users
        similar_users = self.get_similar_users(user_id, n_similar=self.n_neighbors)
        
        if not similar_users:
            return []
        
        # Calculate weighted recommendations
        recommendations = defaultdict(float)
        total_similarity = sum(similarity for _, similarity in similar_users)
        
        reverse_item_index = {v: k for k, v in self.item_index.items()}
        
        for similar_user_id, similarity in similar_users:
            similar_user_idx = self.user_index[similar_user_id]
            similar_user_ratings = self.user_item_matrix[similar_user_idx]
            
            # Recommend items that similar users rated highly but current user hasn't tried
            for item_idx, rating in enumerate(similar_user_ratings):
                if user_ratings[item_idx] == 0 and rating > 0:  # User hasn't tried this intervention
                    intervention_id = reverse_item_index[item_idx]
                    recommendations[intervention_id] += (similarity * rating) / total_similarity
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]

class PersonalizationEngine:
    """
    Advanced personalization engine for user-specific response optimization.
    
    This engine combines deep learning, collaborative filtering, and rule-based
    personalization to create highly tailored therapeutic experiences.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Neural network components
        self.personalization_network = DeepPersonalizationNetwork()
        self.optimizer = torch.optim.Adam(self.personalization_network.parameters(), lr=0.001)
        
        # Collaborative filtering
        self.collaborative_filter = CollaborativeFilteringEngine(
            n_neighbors=self.config.get('cf_neighbors', 10),
            similarity_threshold=self.config.get('cf_similarity_threshold', 0.3)
        )
        
        # Traditional ML components
        self.user_clusterer = KMeans(n_clusters=8, random_state=42)
        self.feature_scaler = StandardScaler()
        self.preference_scaler = MinMaxScaler()
        
        # User profiles and clustering
        self.personalization_profiles = {}  # user_id -> PersonalizationProfile
        self.user_clusters = {}  # cluster_id -> UserCluster
        self.user_to_cluster = {}  # user_id -> cluster_id
        
        # Personalization rules and patterns
        self.personalization_rules = []
        self.intervention_effectiveness_matrix = defaultdict(lambda: defaultdict(float))
        self.user_intervention_ratings = defaultdict(lambda: defaultdict(float))
        
        # Adaptation parameters
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.05)
        self.min_interactions_for_personalization = self.config.get('min_interactions', 5)
        
        # Performance tracking
        self.personalization_metrics = defaultdict(list)
        self.recommendation_history = deque(maxlen=1000)
        
        # Load pre-trained models
        self._load_pretrained_models()
    
    async def personalize_response(self,
                                 user_id: str,
                                 current_context: Dict[str, Any],
                                 available_interventions: List[str],
                                 base_recommendations: Dict[str, Any] = None) -> PersonalizationRecommendation:
        """
        Generate personalized intervention recommendations for a specific user and context.
        """
        try:
            self.logger.info(f"Generating personalized recommendations for user {user_id}")
            
            # Get or create user profile
            profile = await self._get_or_create_profile(user_id)
            
            # Update profile with current context if needed
            await self._update_profile_context(profile, current_context)
            
            # Extract personalization factors
            personalization_factors = await self._extract_personalization_factors(
                profile, current_context
            )
            
            # Generate recommendations using multiple approaches
            
            # 1. Deep learning-based recommendations
            dl_recommendations = await self._generate_deep_learning_recommendations(
                profile, current_context, available_interventions
            )
            
            # 2. Collaborative filtering recommendations
            cf_recommendations = await self._generate_collaborative_recommendations(
                user_id, available_interventions
            )
            
            # 3. Rule-based personalization
            rule_recommendations = await self._apply_personalization_rules(
                profile, current_context, available_interventions
            )
            
            # 4. Cluster-based recommendations
            cluster_recommendations = await self._generate_cluster_recommendations(
                user_id, available_interventions
            )
            
            # Ensemble and rank recommendations
            final_recommendations = await self._ensemble_recommendations(
                [dl_recommendations, cf_recommendations, rule_recommendations, cluster_recommendations],
                profile, current_context
            )
            
            # Generate contextual adaptations
            contextual_adaptations = await self._generate_contextual_adaptations(
                profile, current_context, final_recommendations
            )
            
            # Calculate timing recommendations
            timing_recommendations = await self._calculate_optimal_timing(
                profile, current_context, final_recommendations
            )
            
            # Generate reasoning explanation
            reasoning = await self._generate_personalization_reasoning(
                profile, personalization_factors, final_recommendations
            )
            
            # Create recommendation object
            recommendation = PersonalizationRecommendation(
                recommendation_id=f"pers_{user_id}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                recommended_interventions=final_recommendations['interventions'],
                intervention_weights=final_recommendations['weights'],
                personalization_factors=personalization_factors,
                confidence=final_recommendations['confidence'],
                expected_effectiveness=final_recommendations['expected_effectiveness'],
                reasoning=reasoning,
                contextual_adaptations=contextual_adaptations,
                timing_recommendations=timing_recommendations,
                fallback_options=final_recommendations.get('fallback_options', []),
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=24)
            )
            
            # Store recommendation for learning
            self.recommendation_history.append(recommendation)
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error in personalization: {str(e)}")
            # Return fallback recommendation
            return PersonalizationRecommendation(
                recommendation_id=f"fallback_{user_id}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                recommended_interventions=available_interventions[:3] if available_interventions else [],
                intervention_weights={intervention: 1.0 for intervention in available_interventions[:3]},
                personalization_factors={'error': str(e)},
                confidence=0.0,
                expected_effectiveness=0.5,
                reasoning="Fallback recommendation due to personalization error",
                contextual_adaptations={},
                timing_recommendations={'immediate': True},
                fallback_options=[],
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
    
    async def _get_or_create_profile(self, user_id: str) -> PersonalizationProfile:
        """Get existing user profile or create a new one"""
        
        if user_id in self.personalization_profiles:
            return self.personalization_profiles[user_id]
        
        # Create new profile
        profile = PersonalizationProfile(
            user_id=user_id,
            demographic_features={},
            behavioral_features={},
            preference_vector=np.random.uniform(-0.1, 0.1, 50),  # Random initialization
            response_patterns={},
            intervention_effectiveness={},
            learning_style='adaptive',
            engagement_preferences={},
            temporal_preferences={},
            cultural_context={},
            personality_traits={},
            therapeutic_goals=[],
            contraindications=[],
            adaptation_rate=0.1,
            confidence_scores={},
            last_updated=datetime.now(),
            profile_version=1
        )
        
        self.personalization_profiles[user_id] = profile
        self.logger.info(f"Created new personalization profile for user {user_id}")
        
        return profile
    
    async def _update_profile_context(self, profile: PersonalizationProfile, context: Dict[str, Any]) -> None:
        """Update user profile with current context information"""
        
        # Update behavioral features
        if 'emotional_state' in context:
            profile.behavioral_features['current_emotional_state'] = context['emotional_state']
        
        if 'stress_level' in context:
            profile.behavioral_features['current_stress_level'] = context['stress_level']
        
        if 'session_number' in context:
            profile.behavioral_features['session_count'] = context['session_number']
        
        # Update temporal preferences
        current_time = datetime.now()
        profile.temporal_preferences['last_interaction_hour'] = current_time.hour
        profile.temporal_preferences['last_interaction_day'] = current_time.weekday()
        
        profile.last_updated = current_time
        profile.profile_version += 1
    
    async def _extract_personalization_factors(self, 
                                             profile: PersonalizationProfile,
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key personalization factors from profile and context"""
        
        factors = {
            'user_maturity': len(profile.response_patterns),
            'preference_strength': np.linalg.norm(profile.preference_vector),
            'intervention_history': len(profile.intervention_effectiveness),
            'adaptation_capability': profile.adaptation_rate,
            'profile_confidence': np.mean(list(profile.confidence_scores.values())) if profile.confidence_scores else 0.0,
            'contextual_factors': {
                'emotional_stability': context.get('emotional_state') in ['stable', 'positive'],
                'stress_level': context.get('stress_level', 0.5),
                'time_of_day': datetime.now().hour,
                'session_context': context.get('session_type', 'regular')
            },
            'personalization_readiness': self._calculate_personalization_readiness(profile, context)
        }
        
        return factors
    
    def _calculate_personalization_readiness(self, 
                                           profile: PersonalizationProfile,
                                           context: Dict[str, Any]) -> float:
        """Calculate how ready the user is for personalized interventions"""
        
        readiness_factors = []
        
        # Data availability factor
        data_factor = min(1.0, len(profile.intervention_effectiveness) / self.min_interactions_for_personalization)
        readiness_factors.append(data_factor)
        
        # Profile confidence factor
        confidence_factor = np.mean(list(profile.confidence_scores.values())) if profile.confidence_scores else 0.0
        readiness_factors.append(confidence_factor)
        
        # Contextual stability factor
        stability_factor = 1.0 if context.get('emotional_state') in ['stable', 'motivated'] else 0.5
        readiness_factors.append(stability_factor)
        
        # Engagement factor
        engagement_factor = profile.behavioral_features.get('avg_engagement', 0.5)
        readiness_factors.append(engagement_factor)
        
        return np.mean(readiness_factors)
    
    async def _generate_deep_learning_recommendations(self,
                                                    profile: PersonalizationProfile,
                                                    context: Dict[str, Any],
                                                    available_interventions: List[str]) -> Dict[str, Any]:
        """Generate recommendations using deep personalization network"""
        
        try:
            # Prepare input features
            user_features = self._encode_user_features(profile)
            context_features = self._encode_context_features(context)
            
            # Create intervention mappings
            intervention_vocab = {intervention: i for i, intervention in enumerate(available_interventions)}
            
            recommendations = {}
            
            with torch.no_grad():
                for intervention in available_interventions:
                    # Encode intervention
                    intervention_id = torch.tensor([intervention_vocab[intervention]], dtype=torch.long)
                    user_tensor = torch.FloatTensor(user_features).unsqueeze(0)
                    context_tensor = torch.FloatTensor(context_features).unsqueeze(0)
                    
                    # Forward pass
                    outputs = self.personalization_network(user_tensor, context_tensor, intervention_id)
                    
                    effectiveness = outputs['effectiveness_prediction'].item()
                    preference = outputs['preference_prediction'].item()
                    engagement = outputs['engagement_prediction'].item()
                    
                    # Combined score
                    combined_score = 0.4 * effectiveness + 0.3 * preference + 0.3 * engagement
                    recommendations[intervention] = combined_score
            
            # Rank recommendations
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'source': 'deep_learning',
                'interventions': [item[0] for item in sorted_recommendations],
                'weights': dict(sorted_recommendations),
                'confidence': 0.8,  # High confidence in deep learning
                'expected_effectiveness': np.mean([score for _, score in sorted_recommendations])
            }
            
        except Exception as e:
            self.logger.error(f"Error in deep learning recommendations: {str(e)}")
            return {
                'source': 'deep_learning',
                'interventions': available_interventions[:3],
                'weights': {intervention: 0.5 for intervention in available_interventions[:3]},
                'confidence': 0.0,
                'expected_effectiveness': 0.5
            }
    
    def _encode_user_features(self, profile: PersonalizationProfile) -> np.ndarray:
        """Encode user profile into feature vector for neural network"""
        
        features = np.zeros(100)  # Fixed size feature vector
        
        # Preference vector (first 50 features)
        features[:50] = profile.preference_vector[:50] if len(profile.preference_vector) >= 50 else np.pad(profile.preference_vector, (0, 50 - len(profile.preference_vector)))
        
        # Behavioral features
        idx = 50
        features[idx] = profile.adaptation_rate
        features[idx + 1] = len(profile.intervention_effectiveness) / 20.0  # Normalized
        features[idx + 2] = profile.behavioral_features.get('avg_engagement', 0.5)
        features[idx + 3] = profile.behavioral_features.get('avg_effectiveness', 0.5)
        
        # Temporal features
        features[idx + 4] = profile.temporal_preferences.get('preferred_hour', 12) / 24.0
        features[idx + 5] = profile.temporal_preferences.get('preferred_day', 3) / 7.0
        
        # Personality traits (simplified)
        personality_offset = idx + 6
        for i, trait in enumerate(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']):
            if personality_offset + i < len(features):
                features[personality_offset + i] = profile.personality_traits.get(trait, 0.5)
        
        # Profile maturity
        features[idx + 11] = min(1.0, profile.profile_version / 100.0)
        
        # Confidence scores
        features[idx + 12] = np.mean(list(profile.confidence_scores.values())) if profile.confidence_scores else 0.0
        
        return features
    
    def _encode_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode context into feature vector for neural network"""
        
        features = np.zeros(75)  # Fixed size context vector
        
        # Emotional state encoding
        emotional_states = ['stable', 'anxious', 'depressed', 'motivated', 'frustrated', 'hopeful']
        current_state = context.get('emotional_state', 'stable')
        if current_state in emotional_states:
            features[emotional_states.index(current_state)] = 1.0
        
        # Stress level
        features[6] = context.get('stress_level', 0.5)
        
        # Time features
        current_time = datetime.now()
        features[7] = current_time.hour / 24.0
        features[8] = current_time.weekday() / 7.0
        features[9] = current_time.day / 31.0
        
        # Session features
        features[10] = context.get('session_number', 1) / 50.0  # Normalized
        features[11] = context.get('session_duration', 30) / 120.0  # Normalized to 2 hours max
        
        # Crisis indicators
        features[12] = float(context.get('crisis_detected', False))
        features[13] = float(context.get('safety_concern', False))
        
        # Engagement indicators
        features[14] = context.get('user_engagement_score', 0.5)
        features[15] = context.get('response_time', 30) / 300.0  # Normalized to 5 minutes max
        
        return features
    
    async def _generate_collaborative_recommendations(self,
                                                    user_id: str,
                                                    available_interventions: List[str]) -> Dict[str, Any]:
        """Generate recommendations using collaborative filtering"""
        
        if not self.collaborative_filter.fitted:
            # Try to fit the model if we have enough data
            if len(self.user_intervention_ratings) >= 3:
                self.collaborative_filter.fit(dict(self.user_intervention_ratings))
            else:
                # Not enough data for collaborative filtering
                return {
                    'source': 'collaborative_filtering',
                    'interventions': available_interventions[:3],
                    'weights': {intervention: 0.5 for intervention in available_interventions[:3]},
                    'confidence': 0.0,
                    'expected_effectiveness': 0.5
                }
        
        # Get recommendations from collaborative filtering
        cf_recommendations = self.collaborative_filter.recommend_interventions(
            user_id, n_recommendations=len(available_interventions)
        )
        
        # Filter to available interventions
        filtered_recommendations = [
            (intervention, score) for intervention, score in cf_recommendations
            if intervention in available_interventions
        ]
        
        if not filtered_recommendations:
            # Fallback to available interventions
            filtered_recommendations = [(intervention, 0.5) for intervention in available_interventions[:3]]
        
        return {
            'source': 'collaborative_filtering',
            'interventions': [item[0] for item in filtered_recommendations],
            'weights': dict(filtered_recommendations),
            'confidence': min(1.0, len(filtered_recommendations) / 3.0),
            'expected_effectiveness': np.mean([score for _, score in filtered_recommendations]) if filtered_recommendations else 0.5
        }
    
    async def _apply_personalization_rules(self,
                                         profile: PersonalizationProfile,
                                         context: Dict[str, Any],
                                         available_interventions: List[str]) -> Dict[str, Any]:
        """Apply rule-based personalization logic"""
        
        recommendations = {}
        
        # Rule 1: Prefer interventions that have worked well for this user
        for intervention in available_interventions:
            base_score = profile.intervention_effectiveness.get(intervention, 0.5)
            recommendations[intervention] = base_score
        
        # Rule 2: Adjust based on current emotional state
        emotional_state = context.get('emotional_state', 'stable')
        
        if emotional_state == 'anxious':
            # Prefer calming interventions
            for intervention in ['mindfulness', 'breathing_exercises', 'grounding_techniques']:
                if intervention in recommendations:
                    recommendations[intervention] += 0.2
        
        elif emotional_state == 'depressed':
            # Prefer activating interventions
            for intervention in ['behavioral_activation', 'pleasant_activities', 'social_connection']:
                if intervention in recommendations:
                    recommendations[intervention] += 0.2
        
        elif emotional_state == 'motivated':
            # Prefer skill-building interventions
            for intervention in ['cbt_techniques', 'problem_solving', 'goal_setting']:
                if intervention in recommendations:
                    recommendations[intervention] += 0.2
        
        # Rule 3: Adjust based on time of day
        current_hour = datetime.now().hour
        
        if 6 <= current_hour <= 10:  # Morning
            for intervention in ['goal_setting', 'planning', 'energy_building']:
                if intervention in recommendations:
                    recommendations[intervention] += 0.1
        
        elif 18 <= current_hour <= 22:  # Evening
            for intervention in ['reflection', 'relaxation', 'mindfulness']:
                if intervention in recommendations:
                    recommendations[intervention] += 0.1
        
        # Rule 4: Consider user's therapeutic goals
        for goal in profile.therapeutic_goals:
            if goal == 'anxiety_management':
                for intervention in ['cbt_for_anxiety', 'exposure_therapy', 'relaxation']:
                    if intervention in recommendations:
                        recommendations[intervention] += 0.15
            elif goal == 'depression_treatment':
                for intervention in ['cbt_for_depression', 'behavioral_activation', 'cognitive_restructuring']:
                    if intervention in recommendations:
                        recommendations[intervention] += 0.15
        
        # Rule 5: Apply contraindications
        for contraindication in profile.contraindications:
            if contraindication in recommendations:
                recommendations[contraindication] = max(0.0, recommendations[contraindication] - 0.5)
        
        # Normalize scores
        if recommendations:
            max_score = max(recommendations.values())
            min_score = min(recommendations.values())
            if max_score > min_score:
                for intervention in recommendations:
                    recommendations[intervention] = (recommendations[intervention] - min_score) / (max_score - min_score)
        
        # Sort recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'source': 'rule_based',
            'interventions': [item[0] for item in sorted_recommendations],
            'weights': dict(sorted_recommendations),
            'confidence': 0.6,  # Moderate confidence in rules
            'expected_effectiveness': np.mean([score for _, score in sorted_recommendations]) if sorted_recommendations else 0.5
        }
    
    async def _generate_cluster_recommendations(self,
                                              user_id: str,
                                              available_interventions: List[str]) -> Dict[str, Any]:
        """Generate recommendations based on user cluster"""
        
        # Get user's cluster
        cluster_id = self.user_to_cluster.get(user_id)
        
        if cluster_id and cluster_id in self.user_clusters:
            cluster = self.user_clusters[cluster_id]
            
            # Recommend interventions that work well for this cluster
            cluster_recommendations = {}
            
            for intervention in available_interventions:
                if intervention in cluster.effective_interventions:
                    # High score for cluster's effective interventions
                    cluster_recommendations[intervention] = 0.8
                else:
                    # Lower score for other interventions
                    cluster_recommendations[intervention] = 0.4
            
            # Adjust based on cluster characteristics
            if 'high_engagement' in cluster.cluster_characteristics:
                for intervention in ['interactive_exercises', 'gamified_interventions']:
                    if intervention in cluster_recommendations:
                        cluster_recommendations[intervention] += 0.1
            
            if 'prefers_structured' in cluster.cluster_characteristics:
                for intervention in ['structured_therapy', 'guided_exercises']:
                    if intervention in cluster_recommendations:
                        cluster_recommendations[intervention] += 0.1
            
            sorted_recommendations = sorted(cluster_recommendations.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'source': 'cluster_based',
                'interventions': [item[0] for item in sorted_recommendations],
                'weights': dict(sorted_recommendations),
                'confidence': cluster.coherence_score,
                'expected_effectiveness': np.mean([score for _, score in sorted_recommendations])
            }
        
        else:
            # User not in any cluster or cluster doesn't exist
            return {
                'source': 'cluster_based',
                'interventions': available_interventions[:3],
                'weights': {intervention: 0.5 for intervention in available_interventions[:3]},
                'confidence': 0.0,
                'expected_effectiveness': 0.5
            }
    
    async def _ensemble_recommendations(self,
                                      recommendation_lists: List[Dict[str, Any]],
                                      profile: PersonalizationProfile,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple recommendation sources using ensemble methods"""
        
        # Filter out invalid recommendations
        valid_recommendations = [rec for rec in recommendation_lists if rec['confidence'] > 0.0]
        
        if not valid_recommendations:
            # Fallback
            return {
                'interventions': [],
                'weights': {},
                'confidence': 0.0,
                'expected_effectiveness': 0.5,
                'fallback_options': []
            }
        
        # Calculate ensemble weights based on confidence and source
        source_weights = {
            'deep_learning': 0.4,
            'collaborative_filtering': 0.3,
            'rule_based': 0.2,
            'cluster_based': 0.1
        }
        
        # Weighted combination of recommendations
        combined_weights = defaultdict(float)
        total_weight = 0.0
        
        for rec in valid_recommendations:
            source = rec['source']
            source_weight = source_weights.get(source, 0.1)
            confidence_weight = rec['confidence']
            
            effective_weight = source_weight * confidence_weight
            total_weight += effective_weight
            
            for intervention, score in rec['weights'].items():
                combined_weights[intervention] += effective_weight * score
        
        # Normalize weights
        if total_weight > 0:
            for intervention in combined_weights:
                combined_weights[intervention] /= total_weight
        
        # Sort by combined weight
        sorted_interventions = sorted(combined_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean([rec['confidence'] for rec in valid_recommendations])
        
        # Calculate expected effectiveness
        expected_effectiveness = np.mean([rec['expected_effectiveness'] for rec in valid_recommendations])
        
        # Get top recommendations and fallback options
        top_interventions = sorted_interventions[:5]
        fallback_options = [item[0] for item in sorted_interventions[5:8]]
        
        return {
            'interventions': [item[0] for item in top_interventions],
            'weights': dict(top_interventions),
            'confidence': ensemble_confidence,
            'expected_effectiveness': expected_effectiveness,
            'fallback_options': fallback_options,
            'ensemble_details': {
                'sources_used': [rec['source'] for rec in valid_recommendations],
                'source_weights': source_weights,
                'total_candidates': len(combined_weights)
            }
        }
    
    async def _generate_contextual_adaptations(self,
                                             profile: PersonalizationProfile,
                                             context: Dict[str, Any],
                                             recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contextual adaptations for the recommendations"""
        
        adaptations = {}
        
        # Adapt based on emotional state
        emotional_state = context.get('emotional_state', 'stable')
        
        if emotional_state == 'crisis':
            adaptations['crisis_mode'] = {
                'simplified_language': True,
                'shortened_interventions': True,
                'increased_support': True,
                'immediate_safety_focus': True
            }
        
        elif emotional_state == 'anxious':
            adaptations['anxiety_adaptations'] = {
                'calming_tone': True,
                'slower_pacing': True,
                'grounding_elements': True,
                'reassurance_focus': True
            }
        
        elif emotional_state == 'depressed':
            adaptations['depression_adaptations'] = {
                'encouraging_tone': True,
                'energy_building': True,
                'small_steps_focus': True,
                'hope_instilling': True
            }
        
        # Adapt based on user preferences
        if profile.learning_style == 'visual':
            adaptations['visual_adaptations'] = {
                'include_imagery': True,
                'visual_exercises': True,
                'diagram_support': True
            }
        
        elif profile.learning_style == 'kinesthetic':
            adaptations['kinesthetic_adaptations'] = {
                'physical_exercises': True,
                'movement_based': True,
                'hands_on_activities': True
            }
        
        # Adapt based on engagement preferences
        if profile.engagement_preferences.get('prefers_interactive', False):
            adaptations['interaction_adaptations'] = {
                'interactive_elements': True,
                'feedback_requests': True,
                'collaborative_approach': True
            }
        
        # Cultural adaptations
        if profile.cultural_context:
            adaptations['cultural_adaptations'] = {
                'culturally_sensitive': True,
                'context_aware': True,
                'respectful_approach': True
            }
        
        # Time-based adaptations
        current_hour = datetime.now().hour
        
        if current_hour < 6 or current_hour > 22:
            adaptations['time_adaptations'] = {
                'brief_interventions': True,
                'low_energy_required': True,
                'quiet_activities': True
            }
        
        return adaptations
    
    async def _calculate_optimal_timing(self,
                                      profile: PersonalizationProfile,
                                      context: Dict[str, Any],
                                      recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing for interventions"""
        
        timing_recommendations = {}
        
        # Analyze user's temporal preferences
        preferred_hours = profile.temporal_preferences.get('active_hours', [9, 10, 11, 14, 15, 16, 19, 20])
        current_hour = datetime.now().hour
        
        if current_hour in preferred_hours:
            timing_recommendations['immediate_suitable'] = True
            timing_recommendations['optimal_window'] = 'current'
        else:
            timing_recommendations['immediate_suitable'] = False
            timing_recommendations['next_optimal_window'] = min([h for h in preferred_hours if h > current_hour], default=min(preferred_hours))
        
        # Consider intervention-specific timing
        intervention_timing = {}
        
        for intervention in recommendations['interventions']:
            if 'mindfulness' in intervention.lower() or 'meditation' in intervention.lower():
                intervention_timing[intervention] = {
                    'best_times': ['morning', 'evening'],
                    'duration_minutes': 10,
                    'frequency': 'daily'
                }
            
            elif 'exercise' in intervention.lower() or 'activity' in intervention.lower():
                intervention_timing[intervention] = {
                    'best_times': ['morning', 'afternoon'],
                    'duration_minutes': 30,
                    'frequency': '3-4 times per week'
                }
            
            elif 'reflection' in intervention.lower() or 'journaling' in intervention.lower():
                intervention_timing[intervention] = {
                    'best_times': ['evening'],
                    'duration_minutes': 15,
                    'frequency': 'daily'
                }
        
        timing_recommendations['intervention_timing'] = intervention_timing
        
        # Consider session scheduling
        last_session = profile.temporal_preferences.get('last_session_time')
        
        if last_session:
            hours_since_last = (datetime.now() - last_session).total_seconds() / 3600
            
            if hours_since_last < 4:
                timing_recommendations['session_spacing'] = 'too_recent'
                timing_recommendations['recommended_delay_hours'] = 4 - hours_since_last
            elif hours_since_last > 48:
                timing_recommendations['session_spacing'] = 'overdue'
                timing_recommendations['urgency'] = 'high'
            else:
                timing_recommendations['session_spacing'] = 'appropriate'
        
        return timing_recommendations
    
    async def _generate_personalization_reasoning(self,
                                                profile: PersonalizationProfile,
                                                factors: Dict[str, Any],
                                                recommendations: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for personalization decisions"""
        
        reasoning_parts = []
        
        # User maturity reasoning
        if factors['user_maturity'] > 10:
            reasoning_parts.append(f"Based on your extensive interaction history ({factors['user_maturity']} sessions)")
        elif factors['user_maturity'] > 3:
            reasoning_parts.append("Based on our growing understanding of your preferences")
        else:
            reasoning_parts.append("As we're still learning about your preferences")
        
        # Effectiveness reasoning
        if recommendations.get('expected_effectiveness', 0) > 0.7:
            reasoning_parts.append("these interventions show high potential effectiveness for you")
        elif recommendations.get('expected_effectiveness', 0) > 0.5:
            reasoning_parts.append("these interventions are moderately well-suited to your profile")
        else:
            reasoning_parts.append("we're recommending exploratory interventions to better understand your needs")
        
        # Context reasoning
        contextual_factors = factors.get('contextual_factors', {})
        
        if not contextual_factors.get('emotional_stability', True):
            reasoning_parts.append("Given your current emotional state, we've prioritized stabilizing interventions")
        
        if contextual_factors.get('stress_level', 0.5) > 0.7:
            reasoning_parts.append("Considering your elevated stress level, we've included stress-reduction techniques")
        
        # Personalization readiness
        readiness = factors.get('personalization_readiness', 0.5)
        
        if readiness > 0.8:
            reasoning_parts.append("Your profile enables highly personalized recommendations")
        elif readiness > 0.5:
            reasoning_parts.append("We're able to provide moderately personalized suggestions")
        else:
            reasoning_parts.append("We're using general best practices while building your personalized profile")
        
        # Ensemble reasoning
        ensemble_details = recommendations.get('ensemble_details', {})
        sources_used = ensemble_details.get('sources_used', [])
        
        if len(sources_used) > 2:
            reasoning_parts.append(f"These recommendations combine insights from {len(sources_used)} different analytical approaches")
        
        # Join reasoning parts
        if len(reasoning_parts) > 1:
            reasoning = ". ".join(reasoning_parts[:-1]) + ", and " + reasoning_parts[-1] + "."
        elif reasoning_parts:
            reasoning = reasoning_parts[0] + "."
        else:
            reasoning = "These recommendations are based on clinical best practices and your available data."
        
        return reasoning
    
    async def learn_from_outcome(self,
                               user_id: str,
                               intervention_id: str,
                               outcome: InterventionOutcome) -> None:
        """Learn from intervention outcomes to improve personalization"""
        
        try:
            # Get user profile
            if user_id not in self.personalization_profiles:
                await self._get_or_create_profile(user_id)
            
            profile = self.personalization_profiles[user_id]
            
            # Update intervention effectiveness
            intervention_type = outcome.intervention_type
            effectiveness = outcome.effectiveness_score
            
            # Adaptive learning rate based on confidence
            current_effectiveness = profile.intervention_effectiveness.get(intervention_type, 0.5)
            learning_rate = profile.adaptation_rate
            
            # Update with exponential moving average
            new_effectiveness = (1 - learning_rate) * current_effectiveness + learning_rate * effectiveness
            profile.intervention_effectiveness[intervention_type] = new_effectiveness
            
            # Update user-intervention ratings for collaborative filtering
            self.user_intervention_ratings[user_id][intervention_type] = new_effectiveness
            
            # Update preference vector based on outcome
            if hasattr(outcome, 'user_response') and outcome.user_response:
                # Analyze user response to update preferences (simplified)
                response_sentiment = self._analyze_response_sentiment(outcome.user_response)
                
                # Update preference vector (simplified approach)
                preference_update = np.random.uniform(-0.05, 0.05, len(profile.preference_vector))
                preference_update *= response_sentiment  # Scale by sentiment
                
                profile.preference_vector += preference_update
                profile.preference_vector = np.clip(profile.preference_vector, -1.0, 1.0)
            
            # Update behavioral features
            profile.behavioral_features['last_effectiveness'] = effectiveness
            profile.behavioral_features['last_engagement'] = outcome.engagement_score
            
            if 'avg_effectiveness' not in profile.behavioral_features:
                profile.behavioral_features['avg_effectiveness'] = effectiveness
            else:
                current_avg = profile.behavioral_features['avg_effectiveness']
                profile.behavioral_features['avg_effectiveness'] = (
                    0.9 * current_avg + 0.1 * effectiveness
                )
            
            # Update confidence scores
            if intervention_type not in profile.confidence_scores:
                profile.confidence_scores[intervention_type] = 0.5
            
            # Increase confidence if outcome matches prediction, decrease if not
            expected_range = (new_effectiveness - 0.1, new_effectiveness + 0.1)
            if expected_range[0] <= effectiveness <= expected_range[1]:
                profile.confidence_scores[intervention_type] = min(1.0, profile.confidence_scores[intervention_type] + 0.1)
            else:
                profile.confidence_scores[intervention_type] = max(0.1, profile.confidence_scores[intervention_type] - 0.05)
            
            # Update temporal preferences if breakthrough occurred
            if outcome.breakthrough_indicator:
                current_hour = outcome.timestamp.hour
                profile.temporal_preferences['breakthrough_hours'] = profile.temporal_preferences.get('breakthrough_hours', [])
                profile.temporal_preferences['breakthrough_hours'].append(current_hour)
            
            # Update profile metadata
            profile.last_updated = datetime.now()
            profile.profile_version += 1
            
            # Store training data for neural network
            if len(self.user_intervention_ratings) % 50 == 0:  # Periodically retrain
                await self._retrain_models()
            
            self.logger.debug(f"Updated personalization profile for user {user_id} based on {intervention_type} outcome")
            
        except Exception as e:
            self.logger.error(f"Error learning from outcome: {str(e)}")
    
    def _analyze_response_sentiment(self, response: str) -> float:
        """Analyze sentiment of user response (-1.0 to 1.0)"""
        
        if not response:
            return 0.0
        
        positive_words = [
            'good', 'great', 'helpful', 'better', 'thanks', 'positive',
            'understand', 'clear', 'makes sense', 'insightful', 'useful'
        ]
        
        negative_words = [
            'bad', 'worse', 'difficult', 'hard', 'negative', 'frustrated',
            'confused', 'unhelpful', 'unclear', 'pointless', 'useless'
        ]
        
        response_lower = response.lower()
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        total_words = len(response_lower.split())
        
        if total_words == 0:
            return 0.0
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        return positive_ratio - negative_ratio
    
    async def _retrain_models(self) -> None:
        """Retrain machine learning models with new data"""
        
        try:
            # Retrain collaborative filtering
            if len(self.user_intervention_ratings) >= 3:
                self.collaborative_filter.fit(dict(self.user_intervention_ratings))
                self.logger.info("Retrained collaborative filtering model")
            
            # Update user clustering
            await self._update_user_clustering()
            
            # Note: Neural network retraining would require more sophisticated implementation
            # For now, we just log that it should be done
            self.logger.info("Personalization models updated with new data")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {str(e)}")
    
    async def _update_user_clustering(self) -> None:
        """Update user clustering based on current profiles"""
        
        if len(self.personalization_profiles) < 3:
            return  # Need at least 3 users for clustering
        
        try:
            # Prepare feature matrix for clustering
            user_features = []
            user_ids = []
            
            for user_id, profile in self.personalization_profiles.items():
                features = self._encode_user_features(profile)
                user_features.append(features)
                user_ids.append(user_id)
            
            user_features = np.array(user_features)
            
            # Standardize features
            standardized_features = self.feature_scaler.fit_transform(user_features)
            
            # Perform clustering
            n_clusters = min(8, len(user_ids) // 2)  # Adaptive number of clusters
            self.user_clusterer.n_clusters = n_clusters
            
            cluster_labels = self.user_clusterer.fit_predict(standardized_features)
            
            # Create cluster objects
            new_clusters = {}
            cluster_members = defaultdict(list)
            
            for user_id, cluster_label in zip(user_ids, cluster_labels):
                self.user_to_cluster[user_id] = str(cluster_label)
                cluster_members[cluster_label].append(user_id)
            
            # Analyze each cluster
            for cluster_id, member_ids in cluster_members.items():
                if len(member_ids) < 2:
                    continue  # Skip small clusters
                
                # Get cluster characteristics
                cluster_profiles = [self.personalization_profiles[uid] for uid in member_ids]
                
                # Calculate common preferences
                common_preferences = {}
                for intervention in set().union(*[p.intervention_effectiveness.keys() for p in cluster_profiles]):
                    effectiveness_scores = [p.intervention_effectiveness.get(intervention, 0.5) for p in cluster_profiles]
                    common_preferences[intervention] = np.mean(effectiveness_scores)
                
                # Find effective interventions (above threshold)
                effective_interventions = [
                    intervention for intervention, score in common_preferences.items()
                    if score > 0.6
                ]
                
                # Calculate cluster characteristics
                cluster_characteristics = {}
                
                # Check for high engagement
                avg_engagement = np.mean([
                    p.behavioral_features.get('avg_engagement', 0.5)
                    for p in cluster_profiles
                ])
                if avg_engagement > 0.7:
                    cluster_characteristics['high_engagement'] = True
                
                # Check for structured preference
                structured_interventions = ['cbt', 'structured_therapy', 'guided_exercises']
                structured_preference = np.mean([
                    common_preferences.get(intervention, 0.5)
                    for intervention in structured_interventions
                ])
                if structured_preference > 0.6:
                    cluster_characteristics['prefers_structured'] = True
                
                # Calculate coherence score (simplified)
                feature_vectors = [self._encode_user_features(p) for p in cluster_profiles]
                coherence_score = 1.0 - np.mean([
                    np.linalg.norm(fv - np.mean(feature_vectors, axis=0))
                    for fv in feature_vectors
                ]) / np.linalg.norm(np.mean(feature_vectors, axis=0))
                
                # Create cluster object
                cluster = UserCluster(
                    cluster_id=str(cluster_id),
                    cluster_name=f"Cluster_{cluster_id}",
                    member_user_ids=set(member_ids),
                    cluster_characteristics=cluster_characteristics,
                    common_preferences=common_preferences,
                    effective_interventions=effective_interventions,
                    cluster_size=len(member_ids),
                    coherence_score=max(0.0, min(1.0, coherence_score)),
                    representative_features=np.mean(feature_vectors, axis=0),
                    last_updated=datetime.now()
                )
                
                new_clusters[str(cluster_id)] = cluster
            
            self.user_clusters = new_clusters
            self.logger.info(f"Updated user clustering: {len(new_clusters)} clusters for {len(user_ids)} users")
            
        except Exception as e:
            self.logger.error(f"Error updating user clustering: {str(e)}")
    
    def _load_pretrained_models(self) -> None:
        """Load pre-trained personalization models"""
        
        model_dir = self.config.get('model_directory', 'src/data/adaptive_learning/personalization_models')
        
        # Try to load neural network
        nn_path = f"{model_dir}/personalization_network.pt"
        if os.path.exists(nn_path):
            try:
                checkpoint = torch.load(nn_path)
                self.personalization_network.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("Loaded pre-trained personalization network")
            except Exception as e:
                self.logger.warning(f"Could not load personalization network: {str(e)}")
        
        # Try to load user profiles
        profiles_path = f"{model_dir}/user_profiles.pkl"
        if os.path.exists(profiles_path):
            try:
                with open(profiles_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    
                # Convert back to PersonalizationProfile objects
                for user_id, profile_data in saved_data.get('profiles', {}).items():
                    profile_data['preference_vector'] = np.array(profile_data['preference_vector'])
                    profile_data['last_updated'] = datetime.fromisoformat(profile_data['last_updated'])
                    self.personalization_profiles[user_id] = PersonalizationProfile(**profile_data)
                
                # Load other data
                self.user_intervention_ratings = defaultdict(lambda: defaultdict(float), saved_data.get('ratings', {}))
                self.user_to_cluster = saved_data.get('user_to_cluster', {})
                
                self.logger.info(f"Loaded {len(self.personalization_profiles)} user profiles")
                
            except Exception as e:
                self.logger.warning(f"Could not load user profiles: {str(e)}")
    
    def save_models(self, model_dir: str = None) -> None:
        """Save personalization models and data"""
        
        if model_dir is None:
            model_dir = self.config.get('model_directory', 'src/data/adaptive_learning/personalization_models')
        
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Save neural network
            torch.save({
                'model_state_dict': self.personalization_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, f"{model_dir}/personalization_network.pt")
            
            # Save user profiles and related data
            profiles_data = {}
            for user_id, profile in self.personalization_profiles.items():
                profile_dict = asdict(profile)
                profile_dict['preference_vector'] = profile.preference_vector.tolist()
                profile_dict['last_updated'] = profile.last_updated.isoformat()
                profiles_data[user_id] = profile_dict
            
            save_data = {
                'profiles': profiles_data,
                'ratings': dict(self.user_intervention_ratings),
                'user_to_cluster': self.user_to_cluster,
                'clusters': {cid: asdict(cluster) for cid, cluster in self.user_clusters.items()}
            }
            
            with open(f"{model_dir}/user_profiles.pkl", 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"Personalization models saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving personalization models: {str(e)}")
    
    def get_personalization_summary(self) -> Dict[str, Any]:
        """Get summary of personalization engine state"""
        
        return {
            'total_users': len(self.personalization_profiles),
            'total_clusters': len(self.user_clusters),
            'collaborative_filtering_fitted': self.collaborative_filter.fitted,
            'avg_profile_maturity': np.mean([
                len(profile.intervention_effectiveness)
                for profile in self.personalization_profiles.values()
            ]) if self.personalization_profiles else 0,
            'high_confidence_users': len([
                profile for profile in self.personalization_profiles.values()
                if np.mean(list(profile.confidence_scores.values())) > 0.7
            ]) if self.personalization_profiles else 0,
            'recent_recommendations': len(self.recommendation_history),
            'personalization_ready_users': len([
                profile for profile in self.personalization_profiles.values()
                if len(profile.intervention_effectiveness) >= self.min_interactions_for_personalization
            ]),
            'cluster_distribution': {
                cid: cluster.cluster_size
                for cid, cluster in self.user_clusters.items()
            }
        }