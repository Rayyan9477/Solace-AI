from typing import Dict, Optional
from .chat_agent import ChatAgent
from .emotion_agent import EmotionAgent
from .safety_agent import SafetyAgent
from .search_agent import SearchAgent
import logging
from datetime import datetime

class AgentOrchestrator:
    def __init__(self, api_key: str):
        """Initialize the agent orchestrator with Claude API key"""
        self.api_key = api_key
        self.chat_agent = ChatAgent(api_key)
        self.emotion_agent = EmotionAgent(api_key)
        self.safety_agent = SafetyAgent(api_key)
        self.search_agent = SearchAgent(api_key)
        
        self.conversation_history = []
        self.emotion_history = []
        self.safety_history = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_message(self, message: str, context: Optional[str] = None) -> Dict:
        """Process a user message through all agents and return a coordinated response"""
        try:
            # Step 1: Analyze emotions
            emotion_data = self.emotion_agent.analyze(
                message,
                history=self.emotion_history[-1] if self.emotion_history else None
            )
            self.emotion_history.append(emotion_data)
            
            # Step 2: Assess safety
            safety_data = self.safety_agent.analyze(
                message,
                emotion_data=emotion_data,
                history=self.safety_history[-1] if self.safety_history else None
            )
            self.safety_history.append(safety_data)
            
            # Step 3: Get relevant search results if needed
            search_results = ""
            if self._should_perform_search(message, emotion_data):
                search_results = self.search_agent.search(message)
            
            # Step 4: Generate response
            response = self.chat_agent.generate_response(
                context=context or "",
                question=message,
                emotion=emotion_data,
                safety=safety_data,
                search_results=search_results
            )
            
            # Step 5: Log interaction
            self._log_interaction(message, emotion_data, safety_data, response)
            
            # Step 6: Prepare result
            result = {
                'response': response,
                'emotion_analysis': emotion_data,
                'safety_assessment': safety_data,
                'timestamp': datetime.now().isoformat(),
                'requires_escalation': safety_data.get('emergency_protocol', False)
            }
            
            # Store in history
            self.conversation_history.append({
                'user_message': message,
                'system_response': result,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            return {
                'response': "I apologize, but I'm having trouble processing your message. Please try again.",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _should_perform_search(self, message: str, emotion_data: Dict) -> bool:
        """Determine if search is needed based on message and emotional context"""
        # Perform search if:
        # 1. Message contains question-like patterns
        # 2. User is seeking information (based on emotion)
        # 3. Safety level allows for information gathering
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'can', 'could']
        has_question = any(indicator in message.lower() for indicator in question_indicators)
        
        information_seeking_emotions = ['confused', 'curious', 'anxious']
        is_seeking_info = any(emotion in emotion_data.get('secondary_emotions', []) 
                            for emotion in information_seeking_emotions)
        
        return has_question or is_seeking_info

    def _log_interaction(self, 
                        message: str, 
                        emotion_data: Dict, 
                        safety_data: Dict, 
                        response: str) -> None:
        """Log the interaction details for monitoring and improvement"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'emotion_data': emotion_data,
            'safety_data': safety_data,
            'system_response': response
        }
        
        # Log critical safety concerns
        if safety_data.get('emergency_protocol', False):
            self.logger.warning(f"Emergency protocol activated: {safety_data.get('concerns', [])}")
            
        # Log low confidence analyses
        if emotion_data.get('confidence') == 'low' or safety_data.get('confidence') == 'low':
            self.logger.warning("Low confidence in analysis", extra=log_entry)
            
        self.logger.info("Interaction processed successfully", extra=log_entry)

    def get_conversation_history(self) -> list:
        """Get the conversation history"""
        return self.conversation_history

    def get_emotional_trends(self) -> Dict:
        """Analyze emotional trends from history"""
        if not self.emotion_history:
            return {}
            
        emotions = [h.get('primary_emotion') for h in self.emotion_history]
        intensities = [h.get('intensity') for h in self.emotion_history]
        
        return {
            'primary_emotions': emotions,
            'intensity_trend': intensities,
            'average_intensity': sum(intensities) / len(intensities),
            'most_common_emotion': max(set(emotions), key=emotions.count)
        }

    def get_safety_summary(self) -> Dict:
        """Get summary of safety assessments"""
        if not self.safety_history:
            return {}
            
        risk_levels = [h.get('risk_level') for h in self.safety_history]
        protocols = [h.get('emergency_protocol') for h in self.safety_history]
        
        return {
            'risk_level_history': risk_levels,
            'emergency_protocols_activated': sum(protocols),
            'current_risk_level': risk_levels[-1] if risk_levels else 'UNKNOWN'
        } 