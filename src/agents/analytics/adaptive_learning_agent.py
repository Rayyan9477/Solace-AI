"""
Advanced Adaptive Learning Agent for System-wide Learning and Optimization

This agent implements comprehensive reinforcement learning capabilities to optimize
the entire Solace-AI system based on user outcomes, intervention effectiveness,
and agent coordination patterns. It extends the existing AdaptiveLearningEngine
with advanced RL algorithms, privacy-preserving learning, and real-time optimization.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import pickle
import os
from concurrent.futures import ThreadPoolExecutor

# Core imports
from ..base.base_agent import BaseAgent
from src.diagnosis.adaptive_learning import AdaptiveLearningEngine, InterventionOutcome, UserProfile, LearningInsight
from src.database.central_vector_db import CentralVectorDB
from src.utils.logger import get_logger
from src.models.llm import get_llm

# ML and RL imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

logger = get_logger(__name__)

@dataclass
class SystemState:
    """Comprehensive system state representation"""
    timestamp: datetime
    active_agents: Dict[str, Dict[str, Any]]
    user_interactions: Dict[str, Any]
    resource_utilization: Dict[str, float]
    performance_metrics: Dict[str, float]
    workflow_efficiency: Dict[str, float]
    user_satisfaction_scores: Dict[str, float]
    intervention_success_rates: Dict[str, float]

@dataclass
class RLAction:
    """Reinforcement learning action representation"""
    action_id: str
    action_type: str  # parameter_adjustment, workflow_optimization, agent_coordination
    target_agent: str
    parameters: Dict[str, Any]
    expected_reward: float
    confidence: float

@dataclass
class RewardSignal:
    """Multi-objective reward signal"""
    therapeutic_effectiveness: float
    user_engagement: float
    workflow_efficiency: float
    resource_efficiency: float
    safety_score: float
    total_reward: float
    normalized_reward: float

class AdvancedRLEngine:
    """
    Advanced Reinforcement Learning Engine using Deep Q-Learning and Policy Gradients
    for system-wide optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # RL hyperparameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.95)  # Discount factor
        self.epsilon = self.config.get('epsilon', 0.1)  # Exploration rate
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.batch_size = self.config.get('batch_size', 32)
        self.memory_size = self.config.get('memory_size', 10000)
        
        # Neural networks
        self.state_dim = 50  # Comprehensive state representation
        self.action_dim = 20  # Possible optimization actions
        
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.policy_network = self._build_policy_network()
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Multi-objective optimization
        self.reward_weights = {
            'therapeutic_effectiveness': 0.4,
            'user_engagement': 0.25,
            'workflow_efficiency': 0.2,
            'resource_efficiency': 0.1,
            'safety_score': 0.05
        }
        
        # Performance tracking
        self.training_metrics = defaultdict(list)
        self.action_history = deque(maxlen=1000)
        
    def _build_q_network(self) -> nn.Module:
        """Build Deep Q-Network for action value estimation"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def _build_policy_network(self) -> nn.Module:
        """Build policy network for continuous action space"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
    
    async def optimize_system_performance(self,
                                        system_state: SystemState,
                                        recent_outcomes: List[InterventionOutcome]) -> Dict[str, Any]:
        """
        Main optimization function using reinforcement learning
        """
        try:
            # Encode system state
            state_vector = await self._encode_system_state(system_state)
            
            # Calculate multi-objective rewards
            reward_signal = self._calculate_reward_signal(recent_outcomes, system_state)
            
            # Select optimization actions
            actions = await self._select_optimization_actions(state_vector, system_state)
            
            # Execute actions and collect results
            execution_results = await self._execute_optimization_actions(actions, system_state)
            
            # Store experience for learning
            await self._store_experience(state_vector, actions, reward_signal, execution_results)
            
            # Update neural networks
            if len(self.memory) >= self.batch_size:
                await self._update_networks()
            
            # Update target network periodically
            if len(self.action_history) % 100 == 0:
                self._update_target_network()
            
            return {
                'optimization_actions': [asdict(action) for action in actions],
                'reward_signal': asdict(reward_signal),
                'execution_results': execution_results,
                'q_network_loss': self.training_metrics['q_loss'][-1] if self.training_metrics['q_loss'] else 0.0,
                'policy_network_loss': self.training_metrics['policy_loss'][-1] if self.training_metrics['policy_loss'] else 0.0,
                'exploration_rate': self.epsilon
            }
            
        except Exception as e:
            self.logger.error(f"Error in system optimization: {str(e)}")
            return {'error': str(e)}
    
    async def _encode_system_state(self, system_state: SystemState) -> torch.Tensor:
        """Encode system state into neural network input tensor"""
        
        # Agent performance features
        agent_features = []
        for agent_name, metrics in system_state.active_agents.items():
            agent_features.extend([
                metrics.get('response_time', 0.0),
                metrics.get('accuracy', 0.0),
                metrics.get('user_satisfaction', 0.0),
                metrics.get('resource_usage', 0.0)
            ])
        
        # Pad or truncate to fixed size
        agent_features = agent_features[:20] + [0.0] * max(0, 20 - len(agent_features))
        
        # Resource utilization features
        resource_features = [
            system_state.resource_utilization.get('cpu', 0.0),
            system_state.resource_utilization.get('memory', 0.0),
            system_state.resource_utilization.get('gpu', 0.0),
            system_state.resource_utilization.get('network', 0.0)
        ]
        
        # Performance metrics features
        performance_features = [
            system_state.performance_metrics.get('avg_response_time', 0.0),
            system_state.performance_metrics.get('throughput', 0.0),
            system_state.performance_metrics.get('error_rate', 0.0),
            system_state.performance_metrics.get('availability', 0.0)
        ]
        
        # User satisfaction features
        user_features = [
            np.mean(list(system_state.user_satisfaction_scores.values())) if system_state.user_satisfaction_scores else 0.0,
            len(system_state.user_satisfaction_scores),
            np.std(list(system_state.user_satisfaction_scores.values())) if len(system_state.user_satisfaction_scores) > 1 else 0.0
        ]
        
        # Intervention success features
        intervention_features = [
            np.mean(list(system_state.intervention_success_rates.values())) if system_state.intervention_success_rates else 0.0,
            len(system_state.intervention_success_rates),
            max(system_state.intervention_success_rates.values()) if system_state.intervention_success_rates else 0.0,
            min(system_state.intervention_success_rates.values()) if system_state.intervention_success_rates else 0.0
        ]
        
        # Time-based features
        time_features = [
            system_state.timestamp.hour / 24.0,  # Hour of day normalized
            system_state.timestamp.weekday() / 7.0,  # Day of week normalized
            (system_state.timestamp.day - 1) / 31.0  # Day of month normalized
        ]
        
        # Workflow efficiency features
        workflow_features = [
            np.mean(list(system_state.workflow_efficiency.values())) if system_state.workflow_efficiency else 0.0,
            len(system_state.workflow_efficiency),
            max(system_state.workflow_efficiency.values()) if system_state.workflow_efficiency else 0.0
        ]
        
        # Combine all features
        all_features = (agent_features + resource_features + performance_features + 
                       user_features + intervention_features + time_features + workflow_features)
        
        # Ensure exactly state_dim features
        all_features = all_features[:self.state_dim] + [0.0] * max(0, self.state_dim - len(all_features))
        
        return torch.tensor(all_features, dtype=torch.float32)
    
    def _calculate_reward_signal(self,
                               recent_outcomes: List[InterventionOutcome],
                               system_state: SystemState) -> RewardSignal:
        """Calculate multi-objective reward signal"""
        
        # Therapeutic effectiveness reward
        effectiveness_scores = [outcome.effectiveness_score for outcome in recent_outcomes if outcome.effectiveness_score is not None]
        therapeutic_reward = np.mean(effectiveness_scores) if effectiveness_scores else 0.5
        
        # User engagement reward
        engagement_scores = [outcome.engagement_score for outcome in recent_outcomes if outcome.engagement_score is not None]
        engagement_reward = np.mean(engagement_scores) if engagement_scores else 0.5
        
        # Workflow efficiency reward
        workflow_reward = np.mean(list(system_state.workflow_efficiency.values())) if system_state.workflow_efficiency else 0.5
        
        # Resource efficiency reward (inverse of resource utilization)
        resource_utilization = np.mean(list(system_state.resource_utilization.values())) if system_state.resource_utilization else 0.5
        resource_reward = max(0.0, 1.0 - resource_utilization)
        
        # Safety reward (based on breakthrough indicators and critical issues)
        breakthrough_count = sum(1 for outcome in recent_outcomes if outcome.breakthrough_indicator)
        safety_reward = min(1.0, breakthrough_count / max(1, len(recent_outcomes)))
        
        # Calculate total weighted reward
        total_reward = (
            self.reward_weights['therapeutic_effectiveness'] * therapeutic_reward +
            self.reward_weights['user_engagement'] * engagement_reward +
            self.reward_weights['workflow_efficiency'] * workflow_reward +
            self.reward_weights['resource_efficiency'] * resource_reward +
            self.reward_weights['safety_score'] * safety_reward
        )
        
        # Normalize reward to [-1, 1] range
        normalized_reward = 2.0 * total_reward - 1.0
        
        return RewardSignal(
            therapeutic_effectiveness=therapeutic_reward,
            user_engagement=engagement_reward,
            workflow_efficiency=workflow_reward,
            resource_efficiency=resource_reward,
            safety_score=safety_reward,
            total_reward=total_reward,
            normalized_reward=normalized_reward
        )
    
    async def _select_optimization_actions(self,
                                         state_vector: torch.Tensor,
                                         system_state: SystemState) -> List[RLAction]:
        """Select optimization actions using epsilon-greedy policy"""
        
        actions = []
        
        # Get Q-values for all possible actions
        with torch.no_grad():
            q_values = self.q_network(state_vector.unsqueeze(0))
            policy_output = self.policy_network(state_vector.unsqueeze(0))
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Random exploration
            action_indices = np.random.choice(self.action_dim, size=min(3, self.action_dim), replace=False)
        else:
            # Greedy exploitation
            action_indices = torch.topk(q_values, min(3, self.action_dim))[1].squeeze().numpy()
            if isinstance(action_indices, np.int64):
                action_indices = [action_indices]
        
        # Convert action indices to concrete actions
        for i, action_idx in enumerate(action_indices):
            action = await self._create_concrete_action(
                action_idx, 
                policy_output[0][action_idx].item(),
                system_state,
                q_values[0][action_idx].item()
            )
            if action:
                actions.append(action)
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        
        return actions
    
    async def _create_concrete_action(self,
                                    action_idx: int,
                                    policy_value: float,
                                    system_state: SystemState,
                                    q_value: float) -> Optional[RLAction]:
        """Convert neural network output to concrete optimization action"""
        
        action_types = [
            "adjust_learning_rate",
            "optimize_agent_allocation",
            "tune_intervention_parameters",
            "adjust_resource_allocation",
            "optimize_workflow_routing",
            "update_personalization_weights",
            "adjust_privacy_parameters",
            "optimize_caching_strategy",
            "tune_coordination_parameters",
            "adjust_response_timing"
        ]
        
        if action_idx >= len(action_types):
            return None
        
        action_type = action_types[action_idx]
        
        # Select target agent based on current performance
        target_agents = list(system_state.active_agents.keys())
        if not target_agents:
            return None
            
        # Select agent with lowest performance for optimization
        agent_performances = {
            agent: metrics.get('user_satisfaction', 0.5)
            for agent, metrics in system_state.active_agents.items()
        }
        target_agent = min(agent_performances.keys(), key=lambda x: agent_performances[x])
        
        # Generate action parameters based on action type and policy value
        parameters = await self._generate_action_parameters(action_type, policy_value, system_state, target_agent)
        
        return RLAction(
            action_id=f"{action_type}_{target_agent}_{int(datetime.now().timestamp())}",
            action_type=action_type,
            target_agent=target_agent,
            parameters=parameters,
            expected_reward=q_value,
            confidence=abs(q_value)
        )
    
    async def _generate_action_parameters(self,
                                        action_type: str,
                                        policy_value: float,
                                        system_state: SystemState,
                                        target_agent: str) -> Dict[str, Any]:
        """Generate specific parameters for each action type"""
        
        parameters = {}
        
        if action_type == "adjust_learning_rate":
            # Adjust learning rate based on policy value
            base_lr = 0.001
            parameters = {
                "new_learning_rate": base_lr * (1.0 + 0.5 * policy_value),
                "adjustment_factor": 1.0 + 0.5 * policy_value
            }
        
        elif action_type == "optimize_agent_allocation":
            # Redistribute workload among agents
            current_load = system_state.active_agents.get(target_agent, {}).get('current_load', 0.5)
            parameters = {
                "load_adjustment": policy_value * 0.2,  # Adjust load by up to 20%
                "target_load": max(0.1, min(0.9, current_load + policy_value * 0.2))
            }
        
        elif action_type == "tune_intervention_parameters":
            # Adjust intervention selection weights
            parameters = {
                "weight_adjustments": {
                    "cbt_weight": 1.0 + policy_value * 0.3,
                    "dbt_weight": 1.0 + policy_value * 0.3,
                    "supportive_weight": 1.0 + policy_value * 0.3
                },
                "personalization_factor": 1.0 + abs(policy_value) * 0.2
            }
        
        elif action_type == "adjust_resource_allocation":
            # Modify resource allocation
            parameters = {
                "cpu_adjustment": policy_value * 0.1,
                "memory_adjustment": policy_value * 0.1,
                "priority_boost": policy_value > 0
            }
        
        elif action_type == "optimize_workflow_routing":
            # Optimize workflow routing decisions
            parameters = {
                "routing_preference": "shortest_queue" if policy_value > 0 else "load_balance",
                "priority_threshold": 0.5 + policy_value * 0.3
            }
        
        elif action_type == "update_personalization_weights":
            # Update user personalization weights
            parameters = {
                "personalization_strength": max(0.1, min(1.0, 0.5 + policy_value * 0.4)),
                "adaptation_rate": max(0.01, min(0.2, 0.1 + abs(policy_value) * 0.1))
            }
        
        elif action_type == "adjust_privacy_parameters":
            # Adjust privacy-utility tradeoff
            parameters = {
                "privacy_level": max(0.5, min(1.0, 0.8 + policy_value * 0.2)),
                "noise_magnitude": max(0.01, min(0.1, 0.05 + abs(policy_value) * 0.05))
            }
        
        elif action_type == "optimize_caching_strategy":
            # Optimize caching parameters
            parameters = {
                "cache_size_factor": 1.0 + policy_value * 0.5,
                "ttl_adjustment": policy_value * 0.3,
                "eviction_strategy": "lru" if policy_value > 0 else "lfu"
            }
        
        elif action_type == "tune_coordination_parameters":
            # Adjust agent coordination parameters
            parameters = {
                "coordination_frequency": max(1, int(5 + policy_value * 3)),
                "consensus_threshold": max(0.5, min(0.95, 0.7 + policy_value * 0.2))
            }
        
        elif action_type == "adjust_response_timing":
            # Optimize response timing
            parameters = {
                "response_delay": max(0.1, min(2.0, 0.5 + policy_value * 0.5)),
                "batch_processing": policy_value > 0.5
            }
        
        return parameters
    
    async def _execute_optimization_actions(self,
                                          actions: List[RLAction],
                                          system_state: SystemState) -> Dict[str, Any]:
        """Execute the selected optimization actions"""
        
        execution_results = {
            'successful_actions': [],
            'failed_actions': [],
            'performance_changes': {}
        }
        
        for action in actions:
            try:
                # Simulate action execution (in real implementation, this would interface with actual system)
                result = await self._simulate_action_execution(action, system_state)
                
                if result['success']:
                    execution_results['successful_actions'].append({
                        'action_id': action.action_id,
                        'action_type': action.action_type,
                        'target_agent': action.target_agent,
                        'result': result
                    })
                    
                    # Track performance changes
                    if action.target_agent not in execution_results['performance_changes']:
                        execution_results['performance_changes'][action.target_agent] = []
                    
                    execution_results['performance_changes'][action.target_agent].append({
                        'metric': action.action_type,
                        'change': result.get('performance_improvement', 0.0)
                    })
                else:
                    execution_results['failed_actions'].append({
                        'action_id': action.action_id,
                        'error': result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                self.logger.error(f"Error executing action {action.action_id}: {str(e)}")
                execution_results['failed_actions'].append({
                    'action_id': action.action_id,
                    'error': str(e)
                })
        
        return execution_results
    
    async def _simulate_action_execution(self, action: RLAction, system_state: SystemState) -> Dict[str, Any]:
        """Simulate the execution of an optimization action (placeholder for real implementation)"""
        
        # In real implementation, this would interface with actual system components
        # For now, we simulate the effects
        
        success_probability = min(0.9, action.confidence)
        success = np.random.random() < success_probability
        
        if success:
            # Simulate performance improvement
            base_improvement = abs(action.expected_reward) * 0.1
            actual_improvement = base_improvement * np.random.uniform(0.8, 1.2)
            
            return {
                'success': True,
                'performance_improvement': actual_improvement,
                'execution_time': np.random.uniform(0.1, 1.0),
                'resource_cost': np.random.uniform(0.01, 0.05)
            }
        else:
            return {
                'success': False,
                'error': f"Action execution failed for {action.action_type}",
                'retry_possible': True
            }
    
    async def _store_experience(self,
                              state_vector: torch.Tensor,
                              actions: List[RLAction],
                              reward_signal: RewardSignal,
                              execution_results: Dict[str, Any]) -> None:
        """Store experience in replay buffer for learning"""
        
        # Calculate actual reward based on execution results
        actual_reward = reward_signal.normalized_reward
        
        # Adjust reward based on action execution success
        successful_actions = len(execution_results['successful_actions'])
        total_actions = successful_actions + len(execution_results['failed_actions'])
        
        if total_actions > 0:
            success_rate = successful_actions / total_actions
            actual_reward *= success_rate
        
        # Store each action as a separate experience
        for action in actions:
            # Create action vector (one-hot encoding + parameters)
            action_vector = self._encode_action(action)
            
            # Store experience tuple (state, action, reward, next_state, done)
            self.memory.append({
                'state': state_vector,
                'action': action_vector,
                'reward': actual_reward,
                'next_state': None,  # Will be filled in next iteration
                'done': False,
                'action_id': action.action_id
            })
            
            # Update action history for analysis
            self.action_history.append({
                'timestamp': datetime.now(),
                'action_type': action.action_type,
                'target_agent': action.target_agent,
                'expected_reward': action.expected_reward,
                'actual_reward': actual_reward,
                'success': action.action_id in [a['action_id'] for a in execution_results['successful_actions']]
            })
    
    def _encode_action(self, action: RLAction) -> torch.Tensor:
        """Encode action into tensor representation"""
        
        # One-hot encode action type
        action_types = [
            "adjust_learning_rate", "optimize_agent_allocation", "tune_intervention_parameters",
            "adjust_resource_allocation", "optimize_workflow_routing", "update_personalization_weights",
            "adjust_privacy_parameters", "optimize_caching_strategy", "tune_coordination_parameters",
            "adjust_response_timing"
        ]
        
        action_one_hot = [0.0] * len(action_types)
        if action.action_type in action_types:
            action_one_hot[action_types.index(action.action_type)] = 1.0
        
        # Encode parameters (simplified)
        param_values = []
        for key, value in action.parameters.items():
            if isinstance(value, (int, float)):
                param_values.append(float(value))
            elif isinstance(value, bool):
                param_values.append(float(value))
        
        # Pad or truncate to fixed size
        param_values = param_values[:10] + [0.0] * max(0, 10 - len(param_values))
        
        # Combine one-hot encoding and parameters
        action_vector = action_one_hot + param_values
        
        return torch.tensor(action_vector, dtype=torch.float32)
    
    async def _update_networks(self) -> None:
        """Update Q-network and policy network using experience replay"""
        
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        # Prepare batch tensors
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.stack([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        
        # Update Q-network
        current_q_values = self.q_network(states)
        target_q_values = rewards  # Simplified - in full implementation, would use next states
        
        q_loss = nn.MSELoss()(current_q_values.mean(dim=1), target_q_values)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # Update policy network
        policy_outputs = self.policy_network(states)
        policy_loss = -torch.mean(rewards.unsqueeze(1) * policy_outputs)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Store training metrics
        self.training_metrics['q_loss'].append(q_loss.item())
        self.training_metrics['policy_loss'].append(policy_loss.item())
        
        self.logger.debug(f"Network update - Q Loss: {q_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}")
    
    def _update_target_network(self) -> None:
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug("Target network updated")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        return {
            'q_loss_history': list(self.training_metrics['q_loss']),
            'policy_loss_history': list(self.training_metrics['policy_loss']),
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'total_actions_taken': len(self.action_history)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained models to disk"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'policy_network_state_dict': self.policy_network.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'training_metrics': dict(self.training_metrics),
            'epsilon': self.epsilon
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained models from disk"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.training_metrics = defaultdict(list, checkpoint['training_metrics'])
            self.epsilon = checkpoint['epsilon']
            self.logger.info(f"Model loaded from {filepath}")


class AdaptiveLearningAgent(BaseAgent):
    """
    Advanced Adaptive Learning Agent that orchestrates system-wide learning and optimization.
    
    This agent extends the BaseAgent framework to provide comprehensive reinforcement learning
    capabilities for optimizing therapeutic outcomes, agent coordination, and system efficiency.
    """
    
    def __init__(self, model=None, config: Dict[str, Any] = None):
        super().__init__(
            model=model,
            name="adaptive_learning_agent",
            role="System-wide Learning and Optimization Coordinator",
            description="Advanced RL agent for optimizing therapeutic outcomes and system performance",
            instructions="""
            You are the Adaptive Learning Agent responsible for continuously learning and optimizing
            the entire Solace-AI system. Your primary functions include:
            
            1. Monitor system-wide performance metrics and user outcomes
            2. Use reinforcement learning to optimize agent performance
            3. Identify patterns in therapeutic interventions and their effectiveness
            4. Personalize system behavior based on user preferences and outcomes
            5. Optimize workflow coordination and resource utilization
            6. Maintain privacy and security while learning from interactions
            7. Provide actionable insights for system improvement
            
            You should be proactive in identifying optimization opportunities and implementing
            improvements that enhance therapeutic outcomes and user satisfaction.
            """
        )
        
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Initialize core components
        self.rl_engine = AdvancedRLEngine(config.get('rl_config', {}))
        self.adaptive_learning = AdaptiveLearningEngine()
        
        # System state tracking
        self.current_system_state = None
        self.system_state_history = deque(maxlen=100)
        self.performance_baselines = {}
        
        # Learning and optimization settings
        self.optimization_interval = config.get('optimization_interval', 300)  # 5 minutes
        self.min_outcomes_for_optimization = config.get('min_outcomes_for_optimization', 10)
        self.learning_enabled = config.get('learning_enabled', True)
        
        # Performance tracking
        self.optimization_history = deque(maxlen=1000)
        self.performance_improvements = defaultdict(list)
        
        # Initialize with saved models if available
        model_path = config.get('model_path', 'src/data/adaptive_learning/rl_models.pt')
        if os.path.exists(model_path):
            try:
                self.rl_engine.load_model(model_path)
                self.logger.info("Loaded pre-trained RL models")
            except Exception as e:
                self.logger.warning(f"Could not load pre-trained models: {str(e)}")
        
        # Start background optimization task
        self._start_background_optimization()
    
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process optimization requests and provide system insights.
        """
        try:
            self.logger.info(f"Processing adaptive learning request: {query[:100]}...")
            
            # Parse query to understand request type
            request_type = self._parse_request_type(query, context)
            
            if request_type == "optimize_system":
                return await self._handle_system_optimization(context)
            elif request_type == "analyze_patterns":
                return await self._handle_pattern_analysis(context)
            elif request_type == "get_recommendations":
                return await self._handle_recommendation_request(context)
            elif request_type == "performance_report":
                return await self._handle_performance_report(context)
            else:
                # General adaptive learning query
                return await self._handle_general_query(query, context)
                
        except Exception as e:
            self.logger.error(f"Error processing adaptive learning request: {str(e)}")
            return {
                "error": str(e),
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def _parse_request_type(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Parse the query to determine the type of request"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["optimize", "optimization", "improve", "performance"]):
            return "optimize_system"
        elif any(word in query_lower for word in ["pattern", "analyze", "analysis", "trend"]):
            return "analyze_patterns"
        elif any(word in query_lower for word in ["recommend", "suggestion", "advice", "best"]):
            return "get_recommendations"
        elif any(word in query_lower for word in ["report", "metrics", "status", "dashboard"]):
            return "performance_report"
        else:
            return "general_query"
    
    async def _handle_system_optimization(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle system-wide optimization requests"""
        
        # Get current system state
        system_state = await self._collect_system_state()
        
        # Get recent intervention outcomes
        recent_outcomes = await self._get_recent_outcomes()
        
        if len(recent_outcomes) < self.min_outcomes_for_optimization:
            return {
                "message": f"Insufficient data for optimization. Need at least {self.min_outcomes_for_optimization} recent outcomes.",
                "current_outcomes": len(recent_outcomes),
                "optimization_scheduled": False,
                "status": "waiting_for_data"
            }
        
        # Perform RL-based optimization
        optimization_result = await self.rl_engine.optimize_system_performance(
            system_state, recent_outcomes
        )
        
        # Store optimization results
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'system_state': asdict(system_state),
            'optimization_result': optimization_result,
            'outcomes_analyzed': len(recent_outcomes)
        })
        
        return {
            "optimization_completed": True,
            "optimization_result": optimization_result,
            "system_state_summary": self._summarize_system_state(system_state),
            "outcomes_analyzed": len(recent_outcomes),
            "next_optimization": (datetime.now() + timedelta(seconds=self.optimization_interval)).isoformat(),
            "status": "success"
        }
    
    async def _handle_pattern_analysis(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle pattern analysis requests"""
        
        # Get analysis parameters from context
        user_id = context.get('user_id') if context else None
        time_period = context.get('time_period', 30) if context else 30
        
        # Use existing adaptive learning engine for pattern analysis
        pattern_analysis = await self.adaptive_learning.analyze_intervention_patterns(
            user_id=user_id,
            time_period=time_period
        )
        
        # Enhance with RL insights
        rl_insights = self._extract_rl_insights()
        
        return {
            "pattern_analysis": pattern_analysis,
            "rl_insights": rl_insights,
            "analysis_period_days": time_period,
            "user_specific": user_id is not None,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    
    async def _handle_recommendation_request(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle requests for optimization recommendations"""
        
        user_id = context.get('user_id') if context else None
        current_context = context.get('current_context', {}) if context else {}
        
        recommendations = []
        
        # Get personalized recommendations if user_id provided
        if user_id:
            available_interventions = current_context.get('available_interventions', [
                'CBT', 'DBT', 'supportive', 'psychoeducation', 'mindfulness'
            ])
            
            personalized_rec = await self.adaptive_learning.get_personalized_recommendation(
                user_id=user_id,
                current_context=current_context,
                available_interventions=available_interventions
            )
            
            recommendations.append({
                'type': 'personalized_intervention',
                'recommendation': personalized_rec
            })
        
        # Add system-wide optimization recommendations
        system_recommendations = await self._generate_system_recommendations()
        recommendations.extend(system_recommendations)
        
        return {
            "recommendations": recommendations,
            "user_specific": user_id is not None,
            "context_provided": bool(current_context),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    
    async def _handle_performance_report(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle performance reporting requests"""
        
        # Get training metrics from RL engine
        training_metrics = self.rl_engine.get_training_metrics()
        
        # Get system performance baselines
        performance_summary = self._calculate_performance_improvements()
        
        # Get optimization history summary
        optimization_summary = self._summarize_optimization_history()
        
        return {
            "training_metrics": training_metrics,
            "performance_improvements": performance_summary,
            "optimization_history": optimization_summary,
            "current_learning_rate": self.rl_engine.epsilon,
            "total_optimizations": len(self.optimization_history),
            "status": "success"
        }
    
    async def _handle_general_query(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle general adaptive learning queries using LLM"""
        
        # Prepare context for LLM
        system_context = {
            "current_system_state": asdict(self.current_system_state) if self.current_system_state else {},
            "recent_optimizations": len(self.optimization_history),
            "learning_enabled": self.learning_enabled,
            "training_metrics": self.rl_engine.get_training_metrics()
        }
        
        # Use parent class LLM processing with enhanced context
        enhanced_context = {**(context or {}), **system_context}
        
        try:
            llm_response = await super().process(query, enhanced_context)
            
            # Enhance LLM response with adaptive learning insights
            llm_response["adaptive_insights"] = {
                "system_optimization_active": self.learning_enabled,
                "last_optimization": self.optimization_history[-1]['timestamp'].isoformat() if self.optimization_history else None,
                "performance_trend": self._get_performance_trend(),
                "learning_recommendations": await self._generate_learning_recommendations()
            }
            
            return llm_response
            
        except Exception as e:
            self.logger.error(f"Error in LLM processing: {str(e)}")
            return {
                "message": "I'm the Adaptive Learning Agent. I can help optimize system performance, analyze patterns, and provide recommendations for improving therapeutic outcomes.",
                "capabilities": [
                    "System-wide performance optimization",
                    "Intervention pattern analysis", 
                    "Personalized recommendations",
                    "Agent coordination optimization",
                    "Performance monitoring and reporting"
                ],
                "status": "fallback_response"
            }
    
    async def _collect_system_state(self) -> SystemState:
        """Collect comprehensive system state information"""
        
        # In a real implementation, this would interface with actual system components
        # For now, we simulate system state collection
        
        current_time = datetime.now()
        
        # Simulate agent performance metrics
        active_agents = {
            "diagnosis_agent": {
                "response_time": np.random.uniform(0.5, 2.0),
                "accuracy": np.random.uniform(0.7, 0.95),
                "user_satisfaction": np.random.uniform(0.6, 0.9),
                "resource_usage": np.random.uniform(0.3, 0.8),
                "current_load": np.random.uniform(0.2, 0.9)
            },
            "therapy_agent": {
                "response_time": np.random.uniform(0.3, 1.5),
                "accuracy": np.random.uniform(0.8, 0.96),
                "user_satisfaction": np.random.uniform(0.7, 0.95),
                "resource_usage": np.random.uniform(0.2, 0.7),
                "current_load": np.random.uniform(0.1, 0.8)
            },
            "emotion_agent": {
                "response_time": np.random.uniform(0.2, 1.0),
                "accuracy": np.random.uniform(0.75, 0.92),
                "user_satisfaction": np.random.uniform(0.65, 0.88),
                "resource_usage": np.random.uniform(0.25, 0.6),
                "current_load": np.random.uniform(0.15, 0.7)
            }
        }
        
        # Simulate system metrics
        resource_utilization = {
            "cpu": np.random.uniform(0.3, 0.8),
            "memory": np.random.uniform(0.4, 0.75),
            "gpu": np.random.uniform(0.2, 0.6),
            "network": np.random.uniform(0.1, 0.4)
        }
        
        performance_metrics = {
            "avg_response_time": np.mean([agent["response_time"] for agent in active_agents.values()]),
            "throughput": np.random.uniform(50, 200),  # requests per minute
            "error_rate": np.random.uniform(0.01, 0.05),
            "availability": np.random.uniform(0.98, 0.999)
        }
        
        # Simulate user satisfaction and workflow efficiency
        user_satisfaction_scores = {
            f"user_{i}": np.random.uniform(0.6, 0.95) for i in range(1, 21)
        }
        
        workflow_efficiency = {
            "intake_workflow": np.random.uniform(0.7, 0.95),
            "assessment_workflow": np.random.uniform(0.75, 0.9),
            "intervention_workflow": np.random.uniform(0.8, 0.96),
            "follow_up_workflow": np.random.uniform(0.65, 0.88)
        }
        
        intervention_success_rates = {
            "CBT": np.random.uniform(0.7, 0.9),
            "DBT": np.random.uniform(0.65, 0.85),
            "supportive": np.random.uniform(0.6, 0.8),
            "psychoeducation": np.random.uniform(0.75, 0.9),
            "mindfulness": np.random.uniform(0.7, 0.88)
        }
        
        system_state = SystemState(
            timestamp=current_time,
            active_agents=active_agents,
            user_interactions={
                "total_sessions": np.random.randint(100, 500),
                "active_users": np.random.randint(50, 200),
                "avg_session_duration": np.random.uniform(10, 45)
            },
            resource_utilization=resource_utilization,
            performance_metrics=performance_metrics,
            workflow_efficiency=workflow_efficiency,
            user_satisfaction_scores=user_satisfaction_scores,
            intervention_success_rates=intervention_success_rates
        )
        
        # Update current system state and history
        self.current_system_state = system_state
        self.system_state_history.append(system_state)
        
        return system_state
    
    async def _get_recent_outcomes(self, time_window_hours: int = 24) -> List[InterventionOutcome]:
        """Get recent intervention outcomes for analysis"""
        
        # In real implementation, this would query the database
        # For now, simulate recent outcomes
        
        current_time = datetime.now()
        outcomes = []
        
        for i in range(np.random.randint(15, 50)):  # Random number of recent outcomes
            outcome_time = current_time - timedelta(
                hours=np.random.uniform(0, time_window_hours)
            )
            
            outcome = InterventionOutcome(
                intervention_id=f"intervention_{i}_{int(outcome_time.timestamp())}",
                user_id=f"user_{np.random.randint(1, 100)}",
                intervention_type=np.random.choice(['CBT', 'DBT', 'supportive', 'psychoeducation', 'mindfulness']),
                intervention_content=f"Simulated intervention content {i}",
                context={
                    "emotional_state": np.random.choice(['stable', 'anxious', 'depressed', 'motivated']),
                    "stress_level": np.random.uniform(0.1, 0.9),
                    "session_number": np.random.randint(1, 20)
                },
                timestamp=outcome_time,
                user_response=f"Simulated user response {i}",
                engagement_score=np.random.uniform(0.3, 1.0),
                effectiveness_score=np.random.uniform(0.2, 0.95),
                symptom_change=np.random.uniform(-0.2, 0.8),
                breakthrough_indicator=np.random.random() < 0.1,  # 10% chance of breakthrough
                follow_up_completed=np.random.random() < 0.7,
                long_term_outcome=np.random.uniform(0.4, 0.9) if np.random.random() < 0.5 else None
            )
            
            outcomes.append(outcome)
        
        return outcomes
    
    def _summarize_system_state(self, system_state: SystemState) -> Dict[str, Any]:
        """Create a summary of the system state"""
        
        return {
            "agent_count": len(system_state.active_agents),
            "avg_agent_satisfaction": np.mean([
                metrics["user_satisfaction"] 
                for metrics in system_state.active_agents.values()
            ]),
            "avg_response_time": system_state.performance_metrics["avg_response_time"],
            "system_availability": system_state.performance_metrics["availability"],
            "resource_utilization": {
                "avg": np.mean(list(system_state.resource_utilization.values())),
                "max": max(system_state.resource_utilization.values()),
                "bottleneck": max(system_state.resource_utilization.keys(), 
                                key=lambda x: system_state.resource_utilization[x])
            },
            "workflow_efficiency": {
                "avg": np.mean(list(system_state.workflow_efficiency.values())),
                "best": max(system_state.workflow_efficiency.keys(),
                          key=lambda x: system_state.workflow_efficiency[x]),
                "worst": min(system_state.workflow_efficiency.keys(),
                           key=lambda x: system_state.workflow_efficiency[x])
            }
        }
    
    def _extract_rl_insights(self) -> Dict[str, Any]:
        """Extract insights from reinforcement learning training"""
        
        action_history = list(self.rl_engine.action_history)
        
        if not action_history:
            return {"message": "No RL training data available yet"}
        
        # Analyze action success rates
        action_success_rates = defaultdict(list)
        for action in action_history:
            action_success_rates[action['action_type']].append(action['success'])
        
        success_summary = {
            action_type: np.mean(successes) if successes else 0.0
            for action_type, successes in action_success_rates.items()
        }
        
        # Find most/least successful actions
        best_action = max(success_summary.keys(), key=lambda x: success_summary[x]) if success_summary else None
        worst_action = min(success_summary.keys(), key=lambda x: success_summary[x]) if success_summary else None
        
        # Calculate learning progress
        recent_rewards = [action['actual_reward'] for action in action_history[-20:]] if len(action_history) >= 20 else []
        learning_trend = "improving" if len(recent_rewards) > 5 and np.mean(recent_rewards[-5:]) > np.mean(recent_rewards[:-5]) else "stable"
        
        return {
            "action_success_rates": success_summary,
            "best_performing_action": best_action,
            "worst_performing_action": worst_action,
            "total_actions_taken": len(action_history),
            "learning_trend": learning_trend,
            "avg_recent_reward": np.mean(recent_rewards) if recent_rewards else 0.0
        }
    
    async def _generate_system_recommendations(self) -> List[Dict[str, Any]]:
        """Generate system-wide optimization recommendations"""
        
        recommendations = []
        
        # Analyze current system state for recommendations
        if self.current_system_state:
            # Resource utilization recommendations
            cpu_usage = self.current_system_state.resource_utilization.get('cpu', 0.5)
            if cpu_usage > 0.8:
                recommendations.append({
                    'type': 'resource_optimization',
                    'priority': 'high',
                    'title': 'High CPU Usage Detected',
                    'description': f'CPU utilization is at {cpu_usage:.1%}. Consider scaling resources or optimizing workload distribution.',
                    'actionable_steps': [
                        'Add additional compute instances',
                        'Optimize agent workload distribution',
                        'Implement caching for frequent operations'
                    ]
                })
            
            # Agent performance recommendations
            agent_performances = {
                name: metrics['user_satisfaction']
                for name, metrics in self.current_system_state.active_agents.items()
            }
            
            if agent_performances:
                worst_agent = min(agent_performances.keys(), key=lambda x: agent_performances[x])
                if agent_performances[worst_agent] < 0.7:
                    recommendations.append({
                        'type': 'agent_optimization',
                        'priority': 'medium',
                        'title': f'Low Performance: {worst_agent}',
                        'description': f'{worst_agent} has user satisfaction of {agent_performances[worst_agent]:.2f}. Consider optimization.',
                        'actionable_steps': [
                            'Review agent response patterns',
                            'Adjust intervention selection weights',
                            'Provide additional training data'
                        ]
                    })
            
            # Workflow efficiency recommendations
            workflow_efficiencies = self.current_system_state.workflow_efficiency
            if workflow_efficiencies:
                avg_efficiency = np.mean(list(workflow_efficiencies.values()))
                if avg_efficiency < 0.8:
                    recommendations.append({
                        'type': 'workflow_optimization',
                        'priority': 'medium',
                        'title': 'Workflow Efficiency Below Target',
                        'description': f'Average workflow efficiency is {avg_efficiency:.2f}. Target is 0.8+.',
                        'actionable_steps': [
                            'Analyze workflow bottlenecks',
                            'Optimize agent coordination',
                            'Implement parallel processing where possible'
                        ]
                    })
        
        # RL-based recommendations
        rl_insights = self._extract_rl_insights()
        if rl_insights.get('worst_performing_action'):
            recommendations.append({
                'type': 'learning_optimization',
                'priority': 'low',
                'title': 'Reinforcement Learning Insight',
                'description': f'Action type "{rl_insights["worst_performing_action"]}" has low success rate.',
                'actionable_steps': [
                    'Adjust action parameters',
                    'Increase exploration for this action type',
                    'Review action execution logic'
                ]
            })
        
        return recommendations
    
    def _calculate_performance_improvements(self) -> Dict[str, Any]:
        """Calculate performance improvements over time"""
        
        if len(self.system_state_history) < 2:
            return {"message": "Insufficient historical data for performance calculation"}
        
        # Compare current vs baseline performance
        current_state = self.system_state_history[-1]
        baseline_state = self.system_state_history[0]
        
        improvements = {}
        
        # Agent performance improvements
        agent_improvements = {}
        for agent_name in current_state.active_agents:
            if agent_name in baseline_state.active_agents:
                current_satisfaction = current_state.active_agents[agent_name]['user_satisfaction']
                baseline_satisfaction = baseline_state.active_agents[agent_name]['user_satisfaction']
                improvement = current_satisfaction - baseline_satisfaction
                agent_improvements[agent_name] = {
                    'satisfaction_improvement': improvement,
                    'current_satisfaction': current_satisfaction,
                    'baseline_satisfaction': baseline_satisfaction
                }
        
        improvements['agent_performance'] = agent_improvements
        
        # System-level improvements
        current_avg_response_time = current_state.performance_metrics['avg_response_time']
        baseline_avg_response_time = baseline_state.performance_metrics['avg_response_time']
        
        improvements['system_performance'] = {
            'response_time_improvement': baseline_avg_response_time - current_avg_response_time,
            'availability_improvement': (current_state.performance_metrics['availability'] - 
                                       baseline_state.performance_metrics['availability']),
            'throughput_improvement': (current_state.performance_metrics['throughput'] -
                                     baseline_state.performance_metrics['throughput'])
        }
        
        # Workflow improvements
        workflow_improvements = {}
        for workflow_name in current_state.workflow_efficiency:
            if workflow_name in baseline_state.workflow_efficiency:
                improvement = (current_state.workflow_efficiency[workflow_name] -
                             baseline_state.workflow_efficiency[workflow_name])
                workflow_improvements[workflow_name] = improvement
        
        improvements['workflow_efficiency'] = workflow_improvements
        
        return improvements
    
    def _summarize_optimization_history(self) -> Dict[str, Any]:
        """Summarize optimization history"""
        
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        # Calculate optimization frequency
        if len(self.optimization_history) > 1:
            time_diffs = [
                (self.optimization_history[i]['timestamp'] - self.optimization_history[i-1]['timestamp']).total_seconds()
                for i in range(1, len(self.optimization_history))
            ]
            avg_interval = np.mean(time_diffs)
        else:
            avg_interval = 0
        
        # Count successful vs failed actions
        successful_actions = 0
        failed_actions = 0
        
        for opt in self.optimization_history:
            opt_result = opt.get('optimization_result', {})
            successful_actions += len(opt_result.get('successful_actions', []))
            failed_actions += len(opt_result.get('failed_actions', []))
        
        success_rate = successful_actions / (successful_actions + failed_actions) if (successful_actions + failed_actions) > 0 else 0
        
        return {
            "total_optimizations": len(self.optimization_history),
            "avg_optimization_interval_seconds": avg_interval,
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "action_success_rate": success_rate,
            "last_optimization": self.optimization_history[-1]['timestamp'].isoformat() if self.optimization_history else None,
            "optimization_trend": self._get_optimization_trend()
        }
    
    def _get_performance_trend(self) -> str:
        """Analyze overall performance trend"""
        
        if len(self.system_state_history) < 3:
            return "insufficient_data"
        
        # Analyze user satisfaction trend
        recent_satisfactions = []
        for state in self.system_state_history[-5:]:
            avg_satisfaction = np.mean([
                metrics['user_satisfaction'] 
                for metrics in state.active_agents.values()
            ])
            recent_satisfactions.append(avg_satisfaction)
        
        if len(recent_satisfactions) >= 3:
            trend_slope = np.polyfit(range(len(recent_satisfactions)), recent_satisfactions, 1)[0]
            
            if trend_slope > 0.02:
                return "improving"
            elif trend_slope < -0.02:
                return "declining"
            else:
                return "stable"
        
        return "stable"
    
    def _get_optimization_trend(self) -> str:
        """Analyze optimization effectiveness trend"""
        
        if len(self.optimization_history) < 3:
            return "insufficient_data"
        
        # Analyze success rate trend over recent optimizations
        recent_success_rates = []
        for opt in self.optimization_history[-5:]:
            opt_result = opt.get('optimization_result', {})
            successful = len(opt_result.get('successful_actions', []))
            total = successful + len(opt_result.get('failed_actions', []))
            success_rate = successful / total if total > 0 else 0
            recent_success_rates.append(success_rate)
        
        if len(recent_success_rates) >= 3:
            trend_slope = np.polyfit(range(len(recent_success_rates)), recent_success_rates, 1)[0]
            
            if trend_slope > 0.05:
                return "improving"
            elif trend_slope < -0.05:
                return "declining"
            else:
                return "stable"
        
        return "stable"
    
    async def _generate_learning_recommendations(self) -> List[str]:
        """Generate recommendations for improving the learning system"""
        
        recommendations = []
        
        # Analyze RL training metrics
        training_metrics = self.rl_engine.get_training_metrics()
        
        if training_metrics['epsilon'] < 0.05:
            recommendations.append("Consider increasing exploration rate to discover new optimization strategies")
        
        if len(training_metrics['q_loss_history']) > 10:
            recent_losses = training_metrics['q_loss_history'][-10:]
            if np.mean(recent_losses) > np.mean(training_metrics['q_loss_history'][:-10]):
                recommendations.append("Q-network loss is increasing - consider adjusting learning rate or network architecture")
        
        if training_metrics['memory_size'] < 100:
            recommendations.append("Experience replay buffer has limited data - collect more interaction data")
        
        # System-level recommendations
        if self.current_system_state:
            avg_satisfaction = np.mean([
                metrics['user_satisfaction']
                for metrics in self.current_system_state.active_agents.values()
            ])
            
            if avg_satisfaction < 0.7:
                recommendations.append("Overall user satisfaction is low - focus optimization on user experience improvements")
        
        performance_trend = self._get_performance_trend()
        if performance_trend == "declining":
            recommendations.append("Performance trend is declining - review recent optimization actions and adjust strategy")
        
        return recommendations
    
    def _start_background_optimization(self):
        """Start background task for periodic optimization"""
        
        async def optimization_loop():
            while self.learning_enabled:
                try:
                    await asyncio.sleep(self.optimization_interval)
                    
                    # Collect current system state
                    system_state = await self._collect_system_state()
                    
                    # Get recent outcomes
                    recent_outcomes = await self._get_recent_outcomes()
                    
                    # Perform optimization if sufficient data
                    if len(recent_outcomes) >= self.min_outcomes_for_optimization:
                        self.logger.info("Performing scheduled system optimization")
                        
                        optimization_result = await self.rl_engine.optimize_system_performance(
                            system_state, recent_outcomes
                        )
                        
                        # Store optimization results
                        self.optimization_history.append({
                            'timestamp': datetime.now(),
                            'system_state': asdict(system_state),
                            'optimization_result': optimization_result,
                            'outcomes_analyzed': len(recent_outcomes),
                            'scheduled': True
                        })
                        
                        self.logger.info(f"Scheduled optimization completed: {len(optimization_result.get('optimization_actions', []))} actions executed")
                        
                        # Save model periodically
                        if len(self.optimization_history) % 10 == 0:
                            model_path = self.config.get('model_path', 'src/data/adaptive_learning/rl_models.pt')
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            self.rl_engine.save_model(model_path)
                    
                except Exception as e:
                    self.logger.error(f"Error in background optimization: {str(e)}")
        
        # Start the background task
        if self.learning_enabled:
            asyncio.create_task(optimization_loop())
            self.logger.info(f"Background optimization started with {self.optimization_interval}s interval")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the adaptive learning agent"""
        
        return {
            "agent_name": self.name,
            "learning_enabled": self.learning_enabled,
            "optimization_interval_seconds": self.optimization_interval,
            "total_optimizations": len(self.optimization_history),
            "current_exploration_rate": self.rl_engine.epsilon,
            "system_state_history_size": len(self.system_state_history),
            "training_metrics": self.rl_engine.get_training_metrics(),
            "performance_trend": self._get_performance_trend(),
            "last_optimization": self.optimization_history[-1]['timestamp'].isoformat() if self.optimization_history else None,
            "status": "active" if self.learning_enabled else "inactive"
        }