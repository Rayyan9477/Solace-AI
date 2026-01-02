"""
Diagnosis Agent Adapter Service

This module provides backward compatibility adapters for existing diagnosis agents,
allowing them to work seamlessly with the new unified diagnosis system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import asdict

from .interfaces import (
    IDiagnosisAgentAdapter, DiagnosisRequest, DiagnosisResult, 
    DiagnosisType, ConfidenceLevel
)
from src.infrastructure.di.container import Injectable
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiagnosisAgentAdapter(Injectable, IDiagnosisAgentAdapter):
    """
    Adapter service that provides backward compatibility with existing diagnosis agents
    while integrating them with the new unified diagnosis system.
    """
    
    def __init__(self):
        """Initialize the agent adapter."""
        self.logger = get_logger(__name__)
        
        # Supported agent types and their formats
        self.supported_agents = {
            "diagnosis_agent": "basic",
            "comprehensive_diagnosis_agent": "comprehensive", 
            "enhanced_diagnosis_agent": "enhanced",
            "enhanced_integrated_diagnosis_agent": "enhanced_integrated",
            "integrated_diagnosis_agent": "integrated"
        }
        
        # Agent format mappings
        self.format_mappings = {
            "basic": {
                "input_fields": ["text", "symptoms"],
                "output_fields": ["response", "diagnosis", "confidence"]
            },
            "comprehensive": {
                "input_fields": ["text", "history", "context"],
                "output_fields": ["response", "assessment", "recommendations"]
            },
            "enhanced": {
                "input_fields": ["message", "conversation_history", "emotion_data"],
                "output_fields": ["diagnosis_result", "insights", "recommendations"]
            },
            "enhanced_integrated": {
                "input_fields": ["personality_data", "diagnosis_data", "emotion_data"],
                "output_fields": ["integrated_insights", "recommendations", "next_steps"]
            },
            "integrated": {
                "input_fields": ["personality_data", "diagnosis_data"],
                "output_fields": ["integrated_assessment", "recommendations"]
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the adapter service."""
        try:
            self.logger.info("Initializing DiagnosisAgentAdapter")
            
            # Validate format mappings
            self._validate_format_mappings()
            
            self.logger.info("DiagnosisAgentAdapter initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DiagnosisAgentAdapter: {str(e)}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the adapter service."""
        try:
            self.logger.info("Shutting down DiagnosisAgentAdapter")
            # No cleanup needed for this service
            self.logger.info("DiagnosisAgentAdapter shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during DiagnosisAgentAdapter shutdown: {str(e)}")
    
    async def adapt_agent_request(self, 
                                agent_input: Dict[str, Any], 
                                context: Dict[str, Any]) -> DiagnosisRequest:
        """
        Adapt agent input to diagnosis request format.
        
        Args:
            agent_input: Input from existing agent
            context: Agent context
            
        Returns:
            Converted diagnosis request
        """
        try:
            # Extract basic information
            user_id = self._extract_user_id(agent_input, context)
            session_id = self._extract_session_id(agent_input, context)
            message = self._extract_message(agent_input, context)
            conversation_history = self._extract_conversation_history(agent_input, context)
            
            # Extract optional data
            voice_emotion_data = self._extract_voice_emotion_data(agent_input, context)
            personality_data = self._extract_personality_data(agent_input, context)
            cultural_info = self._extract_cultural_info(agent_input, context)
            
            # Determine diagnosis type from agent input or context
            diagnosis_type = self._determine_diagnosis_type(agent_input, context)
            
            # Create diagnosis request
            request = DiagnosisRequest(
                user_id=user_id,
                session_id=session_id,
                message=message,
                conversation_history=conversation_history,
                voice_emotion_data=voice_emotion_data,
                personality_data=personality_data,
                cultural_info=cultural_info,
                context=context,
                diagnosis_type=diagnosis_type
            )
            
            self.logger.debug(f"Adapted agent input to diagnosis request for user {user_id}")
            return request
            
        except Exception as e:
            self.logger.error(f"Error adapting agent request: {str(e)}")
            # Return a minimal valid request
            return DiagnosisRequest(
                user_id=context.get("user_id", "unknown"),
                session_id=context.get("session_id", f"session_{int(datetime.now().timestamp())}"),
                message=str(agent_input.get("message", agent_input.get("text", ""))),
                conversation_history=[],
                diagnosis_type=DiagnosisType.BASIC
            )
    
    async def adapt_diagnosis_response(self, 
                                     diagnosis_result: DiagnosisResult, 
                                     agent_format: str = "default") -> Dict[str, Any]:
        """
        Adapt diagnosis result to agent response format.
        
        Args:
            diagnosis_result: Diagnosis result to adapt
            agent_format: Target agent format
            
        Returns:
            Adapted response for agent consumption
        """
        try:
            # Route to appropriate format adapter
            if agent_format == "basic":
                return self._adapt_to_basic_format(diagnosis_result)
            elif agent_format == "comprehensive":
                return self._adapt_to_comprehensive_format(diagnosis_result)
            elif agent_format == "enhanced":
                return self._adapt_to_enhanced_format(diagnosis_result)
            elif agent_format == "enhanced_integrated":
                return self._adapt_to_enhanced_integrated_format(diagnosis_result)
            elif agent_format == "integrated":
                return self._adapt_to_integrated_format(diagnosis_result)
            else:
                # Default format
                return self._adapt_to_default_format(diagnosis_result)
            
        except Exception as e:
            self.logger.error(f"Error adapting diagnosis response: {str(e)}")
            # Return minimal response
            return {
                "response": "Unable to process diagnosis result",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_supported_agents(self) -> List[str]:
        """
        Get list of supported agent types.
        
        Returns:
            List of supported agent type names
        """
        return list(self.supported_agents.keys())
    
    # Private helper methods for request adaptation
    
    def _extract_user_id(self, agent_input: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Extract user ID from agent input or context."""
        return (
            agent_input.get("user_id") or
            context.get("user_id") or
            context.get("session", {}).get("user_id") or
            "unknown_user"
        )
    
    def _extract_session_id(self, agent_input: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Extract session ID from agent input or context."""
        return (
            agent_input.get("session_id") or
            context.get("session_id") or
            context.get("session", {}).get("session_id") or
            f"session_{int(datetime.now().timestamp())}"
        )
    
    def _extract_message(self, agent_input: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Extract message from agent input."""
        return str(
            agent_input.get("message") or
            agent_input.get("text") or
            agent_input.get("query") or
            agent_input.get("input") or
            ""
        )
    
    def _extract_conversation_history(self, agent_input: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract conversation history from agent input or context."""
        history = (
            agent_input.get("conversation_history") or
            agent_input.get("history") or
            context.get("conversation_history") or
            context.get("history") or
            []
        )
        
        # Ensure it's a list
        if not isinstance(history, list):
            return []
        
        return history
    
    def _extract_voice_emotion_data(self, agent_input: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract voice emotion data from agent input or context."""
        return (
            agent_input.get("voice_emotion_data") or
            agent_input.get("emotion_data") or
            context.get("voice_emotion") or
            context.get("emotion_analysis")
        )
    
    def _extract_personality_data(self, agent_input: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract personality data from agent input or context."""
        return (
            agent_input.get("personality_data") or
            agent_input.get("personality") or
            context.get("personality_data") or
            context.get("personality_assessment")
        )
    
    def _extract_cultural_info(self, agent_input: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract cultural information from agent input or context."""
        return (
            agent_input.get("cultural_info") or
            agent_input.get("cultural_context") or
            context.get("cultural_info") or
            context.get("user_profile", {}).get("cultural_background")
        )
    
    def _determine_diagnosis_type(self, agent_input: Dict[str, Any], context: Dict[str, Any]) -> DiagnosisType:
        """Determine diagnosis type from agent input or context."""
        # Check explicit diagnosis type
        diagnosis_type_str = (
            agent_input.get("diagnosis_type") or
            context.get("diagnosis_type") or
            context.get("requested_type")
        )
        
        if diagnosis_type_str:
            try:
                return DiagnosisType(diagnosis_type_str.lower())
            except ValueError:
                pass
        
        # Infer from agent type or other context
        agent_type = context.get("agent_type", "")
        
        if "enhanced_integrated" in agent_type:
            return DiagnosisType.ENHANCED_INTEGRATED
        elif "comprehensive" in agent_type:
            return DiagnosisType.COMPREHENSIVE
        elif "enhanced" in agent_type:
            return DiagnosisType.COMPREHENSIVE
        elif "differential" in agent_type:
            return DiagnosisType.DIFFERENTIAL
        elif "temporal" in agent_type:
            return DiagnosisType.TEMPORAL
        else:
            return DiagnosisType.BASIC
    
    # Private helper methods for response adaptation
    
    def _adapt_to_basic_format(self, diagnosis_result: DiagnosisResult) -> Dict[str, Any]:
        """Adapt to basic agent format."""
        return {
            "response": diagnosis_result.primary_diagnosis or "No specific diagnosis available",
            "diagnosis": diagnosis_result.primary_diagnosis,
            "confidence": diagnosis_result.confidence_score,
            "symptoms": diagnosis_result.symptoms,
            "recommendations": diagnosis_result.recommendations,
            "timestamp": diagnosis_result.timestamp.isoformat()
        }
    
    def _adapt_to_comprehensive_format(self, diagnosis_result: DiagnosisResult) -> Dict[str, Any]:
        """Adapt to comprehensive agent format."""
        return {
            "response": self._build_comprehensive_response(diagnosis_result),
            "assessment": {
                "primary_diagnosis": diagnosis_result.primary_diagnosis,
                "confidence_level": diagnosis_result.confidence_level.value,
                "confidence_score": diagnosis_result.confidence_score,
                "potential_conditions": diagnosis_result.potential_conditions,
                "symptoms": diagnosis_result.symptoms
            },
            "recommendations": diagnosis_result.recommendations,
            "metadata": {
                "processing_time_ms": diagnosis_result.processing_time_ms,
                "warnings": diagnosis_result.warnings,
                "limitations": diagnosis_result.limitations
            },
            "timestamp": diagnosis_result.timestamp.isoformat()
        }
    
    def _adapt_to_enhanced_format(self, diagnosis_result: DiagnosisResult) -> Dict[str, Any]:
        """Adapt to enhanced agent format."""
        return {
            "diagnosis_result": {
                "primary_diagnosis": diagnosis_result.primary_diagnosis,
                "confidence_score": diagnosis_result.confidence_score,
                "potential_conditions": diagnosis_result.potential_conditions,
                "symptoms": diagnosis_result.symptoms
            },
            "insights": diagnosis_result.memory_insights,
            "recommendations": diagnosis_result.recommendations,
            "context_updates": diagnosis_result.context_updates,
            "processing_metadata": {
                "processing_time_ms": diagnosis_result.processing_time_ms,
                "diagnosis_type": diagnosis_result.diagnosis_type.value,
                "warnings": diagnosis_result.warnings,
                "limitations": diagnosis_result.limitations
            },
            "timestamp": diagnosis_result.timestamp.isoformat()
        }
    
    def _adapt_to_enhanced_integrated_format(self, diagnosis_result: DiagnosisResult) -> Dict[str, Any]:
        """Adapt to enhanced integrated agent format."""
        return {
            "integrated_insights": {
                "primary_diagnosis": diagnosis_result.primary_diagnosis,
                "confidence_assessment": {
                    "level": diagnosis_result.confidence_level.value,
                    "score": diagnosis_result.confidence_score
                },
                "comprehensive_analysis": {
                    "symptoms": diagnosis_result.symptoms,
                    "potential_conditions": diagnosis_result.potential_conditions,
                    "memory_insights": diagnosis_result.memory_insights
                },
                "cultural_adaptations": diagnosis_result.context_updates.get("cultural_adaptations", {}),
                "personalized_elements": diagnosis_result.context_updates
            },
            "recommendations": diagnosis_result.recommendations,
            "next_steps": self._extract_next_steps(diagnosis_result),
            "quality_metrics": {
                "processing_time_ms": diagnosis_result.processing_time_ms,
                "confidence_score": diagnosis_result.confidence_score,
                "warnings": diagnosis_result.warnings,
                "limitations": diagnosis_result.limitations
            },
            "timestamp": diagnosis_result.timestamp.isoformat()
        }
    
    def _adapt_to_integrated_format(self, diagnosis_result: DiagnosisResult) -> Dict[str, Any]:
        """Adapt to integrated agent format."""
        return {
            "integrated_assessment": {
                "diagnosis": {
                    "primary": diagnosis_result.primary_diagnosis,
                    "confidence": diagnosis_result.confidence_score,
                    "alternatives": diagnosis_result.potential_conditions
                },
                "symptoms_analysis": diagnosis_result.symptoms,
                "insights": diagnosis_result.memory_insights,
                "context_integration": diagnosis_result.context_updates
            },
            "recommendations": diagnosis_result.recommendations,
            "assessment_quality": {
                "confidence_level": diagnosis_result.confidence_level.value,
                "processing_time_ms": diagnosis_result.processing_time_ms,
                "warnings": diagnosis_result.warnings,
                "limitations": diagnosis_result.limitations
            },
            "timestamp": diagnosis_result.timestamp.isoformat()
        }
    
    def _adapt_to_default_format(self, diagnosis_result: DiagnosisResult) -> Dict[str, Any]:
        """Adapt to default agent format."""
        return {
            "response": self._build_comprehensive_response(diagnosis_result),
            "diagnosis_data": asdict(diagnosis_result),
            "success": True,
            "timestamp": diagnosis_result.timestamp.isoformat()
        }
    
    def _build_comprehensive_response(self, diagnosis_result: DiagnosisResult) -> str:
        """Build a comprehensive response text from diagnosis result."""
        response_parts = []
        
        if diagnosis_result.primary_diagnosis:
            response_parts.append(f"Primary Assessment: {diagnosis_result.primary_diagnosis}")
            response_parts.append(f"Confidence Level: {diagnosis_result.confidence_level.value} ({diagnosis_result.confidence_score:.1%})")
        
        if diagnosis_result.symptoms:
            response_parts.append(f"Identified Symptoms: {', '.join(diagnosis_result.symptoms)}")
        
        if diagnosis_result.potential_conditions:
            conditions = [f"{c.get('condition', 'Unknown')} ({c.get('confidence', 0):.1%})" 
                         for c in diagnosis_result.potential_conditions[:3]]
            response_parts.append(f"Alternative Considerations: {', '.join(conditions)}")
        
        if diagnosis_result.recommendations:
            response_parts.append("Recommendations:")
            for i, rec in enumerate(diagnosis_result.recommendations[:3], 1):
                response_parts.append(f"  {i}. {rec}")
        
        return "\n\n".join(response_parts) if response_parts else "Assessment completed with limited findings."
    
    def _extract_next_steps(self, diagnosis_result: DiagnosisResult) -> List[str]:
        """Extract next steps from diagnosis result."""
        next_steps = []
        
        # Add immediate recommendations as next steps
        for rec in diagnosis_result.recommendations[:2]:
            next_steps.append(rec)
        
        # Add follow-up based on confidence level
        if diagnosis_result.confidence_level == ConfidenceLevel.LOW:
            next_steps.append("Consider additional assessment or professional consultation")
        elif diagnosis_result.confidence_level == ConfidenceLevel.HIGH:
            next_steps.append("Monitor progress and symptoms over time")
        
        return next_steps
    
    def _validate_format_mappings(self) -> None:
        """Validate that format mappings are correctly configured."""
        for agent_type, format_name in self.supported_agents.items():
            if format_name not in self.format_mappings:
                raise ValueError(f"Format mapping missing for {format_name} (used by {agent_type})")
    
    def get_format_mapping(self, agent_format: str) -> Optional[Dict[str, Any]]:
        """Get format mapping for a specific agent format."""
        return self.format_mappings.get(agent_format)
    
    def is_agent_supported(self, agent_type: str) -> bool:
        """Check if an agent type is supported."""
        return agent_type in self.supported_agents