"""
Comprehensive Diagnosis Agent

This agent integrates the advanced ComprehensiveDiagnosisModule to provide detailed
mental health assessments by combining voice emotion analysis, conversational data,
and personality test results.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from src.agents.base_agent import Agent
from src.diagnosis import ComprehensiveDiagnosisModule, create_diagnosis_module
from src.utils.agentic_rag import AgenticRAG
from src.models.llm import get_llm

# Configure logging
logger = logging.getLogger(__name__)

class ComprehensiveDiagnosisAgent(Agent):
    """
    Agent that provides comprehensive mental health diagnoses by integrating
    multiple data sources with advanced vector search, embeddings, and rule-based reasoning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the comprehensive diagnosis agent
        
        Args:
            config: Configuration options for the agent
        """
        super().__init__(name="comprehensive_diagnosis", description="Comprehensive mental health diagnosis integration")
        self.config = config or {}
        
        # Initialize diagnosis module using factory function
        try:
            self.diagnosis_module = create_diagnosis_module(
                use_agentic_rag=self.config.get("use_agentic_rag", True),
                use_cache=self.config.get("use_vector_cache", True)
            )
            logger.info("Successfully initialized ComprehensiveDiagnosisModule")
        except Exception as e:
            logger.error(f"Error initializing ComprehensiveDiagnosisModule: {e}")
            # Fallback to direct instantiation without factory function
            self.diagnosis_module = ComprehensiveDiagnosisModule()
        
        # Track any persistent user contexts for ongoing assessments
        self.user_contexts = {}
    
    async def process(self, 
                     message: str, 
                     context: Dict[str, Any] = None, 
                     **kwargs) -> Dict[str, Any]:
        """
        Process a message and generate a comprehensive mental health diagnosis
        
        Args:
            message: The user message to process
            context: Additional context information including conversation history,
                     voice emotion data, and personality assessment results
            **kwargs: Additional keyword arguments
            
        Returns:
            Diagnosis results and recommendations
        """
        context = context or {}
        user_id = context.get("user_id", "anonymous")
        session_id = context.get("session_id", "default")
        
        # Initialize or retrieve user context
        user_context_key = f"{user_id}_{session_id}"
        if user_context_key not in self.user_contexts:
            self.user_contexts[user_context_key] = {
                "messages": [],
                "voice_emotions": [],
                "personality_data": None,
                "diagnosis_history": []
            }
        
        user_context = self.user_contexts[user_context_key]
        
        # Add current message to context
        user_context["messages"].append({
            "text": message,
            "timestamp": kwargs.get("timestamp")
        })
        
        # Process any new voice emotion data
        if "voice_emotion" in context:
            user_context["voice_emotions"].append(context["voice_emotion"])
        
        # Update personality data if available
        if "personality" in context:
            user_context["personality_data"] = context["personality"]
        
        # Prepare data for diagnosis
        conversation_data = {
            "text": message,
            "history": user_context["messages"][-10:],  # Use last 10 messages for context
            "extracted_symptoms": context.get("extracted_symptoms", [])
        }
        
        # Prepare voice emotion data
        voice_emotion_data = None
        if user_context["voice_emotions"]:
            # Combine recent voice emotion data
            recent_emotions = user_context["voice_emotions"][-3:]  # Use last 3 emotion readings
            emotions_combined = {}
            characteristics_combined = {}
            
            # Combine emotion scores
            for emotion_data in recent_emotions:
                if "emotions" in emotion_data:
                    for emotion, score in emotion_data["emotions"].items():
                        if emotion not in emotions_combined:
                            emotions_combined[emotion] = []
                        emotions_combined[emotion].append(score)
                
                if "characteristics" in emotion_data:
                    for char, value in emotion_data["characteristics"].items():
                        if char not in characteristics_combined:
                            characteristics_combined[char] = []
                        characteristics_combined[char].append(value)
            
            # Average the scores
            voice_emotion_data = {
                "emotions": {e: sum(scores)/len(scores) for e, scores in emotions_combined.items()},
                "characteristics": {c: sum(values)/len(values) for c, values in characteristics_combined.items()},
                "timestamp": kwargs.get("timestamp")
            }
        
        # Generate diagnosis
        try:
            result = await self.diagnosis_module.generate_diagnosis(
                conversation_data=conversation_data,
                voice_emotion_data=voice_emotion_data,
                personality_data=user_context["personality_data"],
                user_id=user_id,
                session_id=session_id,
                external_context=context.get("external_context")
            )
            
            # Store diagnosis history
            if result["success"]:
                user_context["diagnosis_history"].append({
                    "timestamp": result["timestamp"],
                    "conditions": result["conditions"],
                    "severity": result["severity"],
                    "confidence": result["confidence"]
                })
                
                # Track changes in assessment over time
                if len(user_context["diagnosis_history"]) > 1:
                    previous = user_context["diagnosis_history"][-2]
                    current = user_context["diagnosis_history"][-1]
                    
                    result["assessment_changes"] = {
                        "severity_change": self._compare_severity(previous["severity"], current["severity"]),
                        "confidence_change": round(current["confidence"] - previous["confidence"], 2),
                        "condition_changes": self._compare_conditions(
                            previous.get("conditions", []), 
                            current.get("conditions", [])
                        )
                    }
            
            return {
                "diagnosis_result": result,
                "diagnosis_successful": result["success"],
                "primary_condition": result.get("primary_condition", None),
                "severity": result.get("severity", "none"),
                "confidence": result.get("confidence", 0.0),
                "recommendations": result.get("recommendations", []),
                "response": self._generate_response(result, message, context)
            }
        
        except Exception as e:
            logger.error(f"Error generating diagnosis: {str(e)}")
            return {
                "diagnosis_successful": False,
                "error": str(e),
                "response": "I'm sorry, I encountered an error while analyzing your mental health indicators. Please try again."
            }
    
    def _generate_response(self, 
                          result: Dict[str, Any], 
                          message: str, 
                          context: Dict[str, Any]) -> str:
        """
        Generate a human-friendly response based on the diagnosis results
        
        Args:
            result: The diagnosis result
            message: The original user message
            context: Additional context
            
        Returns:
            Human-friendly response text
        """
        if not result["success"]:
            return ("Based on our conversation, I don't have enough information yet "
                   "to provide a meaningful assessment. As we continue talking, "
                   "I'll be better able to understand your experiences.")
        
        # Get primary condition
        if "conditions" in result and result["conditions"]:
            primary_condition = result["conditions"][0]["name"]
            severity = result["severity"]
            confidence = result["confidence"]
            
            # Format confidence as percentage
            confidence_percent = int(confidence * 100)
            
            # Structure varies based on severity
            if severity == "severe":
                response = (
                    f"Based on our conversation and the patterns I've detected, I'm noticing "
                    f"indicators consistent with {primary_condition} at a concerning level "
                    f"({confidence_percent}% confidence). This suggests that seeking "
                    f"professional support soon might be beneficial. "
                )
            elif severity == "moderate":
                response = (
                    f"I'm noticing some patterns in our conversation that may be associated "
                    f"with {primary_condition} ({confidence_percent}% confidence). "
                    f"These indicators are at a moderate level, which means they're "
                    f"worth paying attention to. "
                )
            else:  # mild
                response = (
                    f"I've noticed some mild indicators in our conversation that have some "
                    f"similarity to patterns associated with {primary_condition} "
                    f"({confidence_percent}% confidence). "
                )
            
            # Add recommendations
            if result["recommendations"]:
                response += "Some steps you might consider include: "
                
                # Only include a few top recommendations to avoid overwhelming
                top_recommendations = result["recommendations"][:3]
                for i, rec in enumerate(top_recommendations):
                    if i == len(top_recommendations) - 1:
                        response += f"and {rec.lower()}."
                    else:
                        response += f"{rec.lower()}, "
                
                response += (
                    "\n\nRemember that I'm not a licensed healthcare provider, and these "
                    "observations are not a clinical diagnosis. If you're concerned about "
                    "your mental health, please speak with a qualified healthcare professional."
                )
            
            return response
        else:
            return (
                "Thank you for sharing. I'm still building an understanding of your experiences. "
                "If you'd like to discuss specific aspects of your mental wellbeing, I'm here to listen."
            )
    
    def _compare_severity(self, previous: str, current: str) -> str:
        """Compare severity levels and return change description"""
        severity_levels = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        if previous not in severity_levels or current not in severity_levels:
            return "unchanged"
        
        diff = severity_levels[current] - severity_levels[previous]
        if diff > 0:
            return "increased"
        elif diff < 0:
            return "decreased"
        else:
            return "unchanged"
    
    def _compare_conditions(self, previous: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare condition lists and identify changes"""
        prev_conditions = {c["name"]: c.get("confidence", 0) for c in previous}
        curr_conditions = {c["name"]: c.get("confidence", 0) for c in current}
        
        result = {
            "new": [],
            "removed": [],
            "confidence_changes": {}
        }
        
        # Find new conditions
        for name in curr_conditions:
            if name not in prev_conditions:
                result["new"].append(name)
            else:
                # Track confidence changes for existing conditions
                change = curr_conditions[name] - prev_conditions[name]
                if abs(change) > 0.05:  # Only report significant changes
                    result["confidence_changes"][name] = round(change, 2)
        
        # Find removed conditions
        for name in prev_conditions:
            if name not in curr_conditions:
                result["removed"].append(name)
        
        return result
        
    async def get_response(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest diagnosis for a user session
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Latest diagnosis result if available
        """
        user_context_key = f"{user_id}_{session_id}"
        if user_context_key in self.user_contexts:
            context = self.user_contexts[user_context_key]
            if context["diagnosis_history"]:
                return context["diagnosis_history"][-1]
        return None