from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from agno.agent import Agent

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrates interactions between multiple specialized agents"""

    def __init__(self, agents: Dict[str, Agent] = None, **named_agents):
        """
        Initialize the agent orchestrator with a dictionary of agents
        
        Can be initialized in two ways:
        1. With a dictionary of agents: AgentOrchestrator(agents={"chat": chat_agent, ...})
        2. With named arguments: AgentOrchestrator(chat=chat_agent, emotion=emotion_agent, ...)

        Args:
            agents: Dictionary of agent instances with keys like 'safety', 'emotion', 'chat', etc.
            **named_agents: Alternative way to provide agents with named arguments
        """
        # Handle both initialization methods
        if agents is None:
            self.agents = named_agents
        else:
            self.agents = agents

        # Backwards compatibility for specific agent access methods
        self._safety_agent = self.agents.get("safety")
        self._emotion_agent = self.agents.get("emotion")
        self._chat_agent = self.agents.get("chat", self.agents.get("chat_agent"))

        self.execution_history = []
        self.conversation_history = []
        self.emotion_history = []
        self.safety_history = []

        logger.info("AgentOrchestrator initialized with %d agents", len(self.agents))

    def add_agent(self, name: str, agent: Agent) -> None:
        """Add an agent to the orchestrator"""
        self.agents[name] = agent
        
        # Update specific agent references if applicable
        if name == "safety":
            self._safety_agent = agent
        elif name == "emotion":
            self._emotion_agent = agent
        elif name in ["chat", "chat_agent"]:
            self._chat_agent = agent

    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the orchestrator"""
        if name in self.agents:
            del self.agents[name]
            return True
        return False

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name"""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agents"""
        return list(self.agents.keys())

    async def process(self,
                     query: str,
                     agent_name: str = None,
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query using a specific agent or delegate to appropriate agent"""
        # Store in execution history
        self.execution_history.append({
            'query': query,
            'agent': agent_name,
            'context': context
        })

        # If agent specified, use that agent
        if agent_name and agent_name in self.agents:
            return await self.agents[agent_name].process(query, context)

        # Otherwise, select the best agent based on query
        selected_agent = self._select_agent(query, context)
        if (selected_agent):
            return await selected_agent.process(query, context)

        # If no agent found, use safe fallback
        return {
            'response': "I'm not sure how to help with that. Could you please rephrase your question?",
            'metadata': {
                'agent_name': 'fallback',
                'confidence': 0.0
            }
        }

    def _select_agent(self, query: str, context: Dict[str, Any] = None) -> Optional[Agent]:
        """Select the most appropriate agent for a query"""
        if not self.agents:
            return None

        # Check if there's an explicit request for a specific agent
        for name, agent in self.agents.items():
            agent_keywords = [
                name.lower(),
                agent.name.lower(),
                agent.description.lower()
            ]
            if any(keyword in query.lower() for keyword in agent_keywords):
                return agent

        # Default to using the chat agent if available
        if 'chat' in self.agents:
            return self.agents['chat']

        # Otherwise return the first available agent
        return list(self.agents.values())[0] if self.agents else None

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user message through all agents and return a coordinated response

        Args:
            message: The user's message
            context: Optional context information

        Returns:
            Dictionary containing the coordinated response
        """
        try:
            # Step 1: Analyze emotions
            emotion_data = await self.agents["emotion"].analyze_emotion(message)
            self.emotion_history.append(emotion_data)

            # Step 2: Assess safety
            safety_data = await self.agents["safety"].check_message(message)
            self.safety_history.append(safety_data)

            # Step 3: Get personality data if available
            personality_data = {}
            if "personality" in self.agents and self.agents["personality"] is not None:
                try:
                    personality_data = await self.agents["personality"].get_previous_assessment()
                except Exception as e:
                    logger.warning(f"Failed to retrieve personality data: {str(e)}")

            # Suggested standardized assessments based on clinical indicators
            suggestions = []
            indicators = emotion_data.get('clinical_indicators', [])
            if 'depression symptoms' in indicators:
                suggestions.append('PHQ-9 assessment suggested to evaluate depression severity')
            if 'anxiety symptoms' in indicators:
                suggestions.append('GAD-7 assessment suggested to evaluate anxiety levels')

            # Step 4: Generate response
            response = await self.agents["chat"].generate_response(
                message=message,
                context={
                    "emotion": emotion_data,
                    "safety": safety_data,
                    "personality": personality_data
                }
            )

            # Step 5: Log interaction
            self._log_interaction(message, emotion_data, safety_data, response)

            # Step 6: Prepare result
            result = {
                'response': response.get('response', ''),
                'emotion_analysis': emotion_data,
                'safety_assessment': safety_data,
                'personality_data': personality_data,
                'assessment_suggestions': suggestions,
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
            logger.error(f"Error processing message: {str(e)}")
            return {
                'response': "I apologize, but I'm having trouble processing your message. Please try again.",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _log_interaction(self,
                        message: str,
                        emotion_data: Dict[str, Any],
                        safety_data: Dict[str, Any],
                        response: Dict[str, Any]) -> None:
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
            logger.warning(f"Emergency protocol activated: {safety_data.get('concerns', [])}")

        # Log low confidence analyses
        if emotion_data.get('confidence', 1.0) < 0.5 or safety_data.get('confidence', 1.0) < 0.5:
            logger.warning("Low confidence in analysis", extra=log_entry)

        logger.info("Interaction processed successfully", extra=log_entry)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history

    def get_emotional_trends(self) -> Dict[str, Any]:
        """Analyze emotional trends from history"""
        if not self.emotion_history:
            return {}

        emotions = [h.get('primary_emotion', 'unknown') for h in self.emotion_history]
        intensities = [h.get('intensity', 5) for h in self.emotion_history]

        return {
            'primary_emotions': emotions,
            'intensity_trend': intensities,
            'average_intensity': sum(intensities) / len(intensities) if intensities else 0,
            'most_common_emotion': max(set(emotions), key=emotions.count) if emotions else 'unknown'
        }

    def get_safety_summary(self) -> Dict[str, Any]:
        """Get summary of safety assessments"""
        if not self.safety_history:
            return {}

        risk_levels = [h.get('risk_level', 'UNKNOWN') for h in self.safety_history]
        protocols = [h.get('emergency_protocol', False) for h in self.safety_history]

        return {
            'risk_level_history': risk_levels,
            'emergency_protocols_activated': sum(1 for p in protocols if p),
            'current_risk_level': risk_levels[-1] if risk_levels else 'UNKNOWN'
        }