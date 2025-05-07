"""
Therapeutic agent for recommending practical steps in responses.

This agent is responsible for analyzing user input and recommending
relevant therapeutic techniques that can be included in the chatbot's responses.
"""

from typing import Dict, List, Optional, Tuple, Any

from src.agents.base_agent import BaseAgent
from src.knowledge.therapeutic.technique_service import TherapeuticTechniqueService

class TherapyAgent(BaseAgent):
    """Agent for recommending therapeutic techniques and practical steps."""
    
    def __init__(self, model_provider=None):
        """Initialize the therapy agent.
        
        Args:
            model_provider: LLM provider for generating embeddings and content
        """
        super().__init__(model_provider)
        self.technique_service = TherapeuticTechniqueService(model_provider)
        
        # Initialize the vector store with therapeutic techniques
        self.technique_service.initialize_vector_store()
        
    def process(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input and recommend therapeutic techniques.
        
        Args:
            user_input: The user's message
            context: Contextual information including conversation history,
                     emotion analysis, etc.
        
        Returns:
            Dictionary containing recommended therapeutic techniques
        """
        # Extract emotion if available in context
        emotion = None
        if context.get("emotion_analysis") and context["emotion_analysis"].get("primary_emotion"):
            emotion = context["emotion_analysis"]["primary_emotion"]
            
        # Get relevant therapeutic techniques based on user input and emotion
        techniques = self.technique_service.get_relevant_techniques(
            query=user_input,
            emotion=emotion,
            top_k=2  # Recommend 2 techniques by default
        )
        
        # Format techniques for inclusion in response
        formatted_techniques = self.technique_service.format_techniques_for_response(techniques)
        
        return {
            "therapeutic_techniques": techniques,
            "formatted_techniques": formatted_techniques
        }
    
    def enhance_response(self, response: str, therapeutic_result: Dict[str, Any]) -> str:
        """Enhance a response with therapeutic techniques.
        
        Args:
            response: Original response from the chatbot
            therapeutic_result: Result from the process method
            
        Returns:
            Enhanced response with therapeutic techniques
        """
        if not therapeutic_result or not therapeutic_result.get("formatted_techniques"):
            return response
            
        # Add therapeutic techniques to the response
        enhanced_response = f"{response}\n\n{therapeutic_result['formatted_techniques']}"
        return enhanced_response