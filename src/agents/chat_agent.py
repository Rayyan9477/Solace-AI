from typing import Optional, Dict, Any, List
# Only Gemini 2.0 Flash is supported. Remove Anthropic/Claude imports.
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.memory import ConversationBufferMemory
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from .base_agent import BaseAgent
import anthropic
import httpx
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        # Remove proxies argument if present
        kwargs.pop("proxies", None)
        super().__init__(*args, **kwargs)

class ChatAgent(BaseAgent):
    def __init__(self, model: BaseLanguageModel):
        # Create a langchain memory instance
        langchain_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        # Create memory dict for agno Memory
        memory_dict = {
            "memory": "chat_memory",  # Memory parameter should be a string
            "storage": "local_storage",  # Storage parameter should be a string
            "memory_key": "chat_history",
            "chat_memory": langchain_memory,
            "input_key": "input",
            "output_key": "output",
            "return_messages": True
        }
        
        # Initialize Memory with the dictionary
        memory = Memory(**memory_dict)
        
        super().__init__(
            model=model,
            name="chat_assistant",
            description="Supportive mental health chat assistant",
            tools=[],
            memory=memory,
            knowledge=AgentKnowledge()
        )
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a supportive mental health assistant.
Your role is to provide empathetic, non-judgmental support while:
1. Maintaining professional boundaries
2. Offering evidence-based information
3. Encouraging professional help when appropriate
4. Avoiding diagnostic statements
5. Focusing on emotional support and coping strategies

Guidelines:
- Be warm and empathetic
- Validate feelings without reinforcing negative patterns
- Suggest practical coping strategies
- Encourage professional help when appropriate
- Maintain a supportive, non-judgmental tone"""),
            HumanMessage(content="""User Message: {message}
Emotional Context: {emotion_data}
Safety Context: {safety_data}
Diagnosis: {diagnosis_data}
Previous Conversation: {history}

Provide a supportive, empathetic response that addresses the user's emotional needs and current diagnosis while maintaining appropriate boundaries and suggesting coping strategies.""")
        ])
    
    async def generate_response(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a supportive response to a user message
        
        Args:
            message: The user's message
            context: Additional context including emotion and safety data
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Get conversation history
            history = await self.memory.get("chat_history", [])
            
            # Format context data
            emotion_data = self._format_emotion_data(context.get("emotion", {})) if context else "No emotional data available"
            safety_data = self._format_safety_data(context.get("safety", {})) if context else "No safety data available"
            # Include diagnosis context
            diagnosis_data = context.get("diagnosis", "No diagnosis available")
            
            # Generate response
            try:
                # Try to use agenerate_messages if available
                llm_response = await self.llm.agenerate_messages([
                    self.chat_prompt.format_messages(
                        message=message,
                        emotion_data=emotion_data,
                        safety_data=safety_data,
                        diagnosis_data=diagnosis_data,
                        history=self._format_history(history)
                    )[0]
                ])
                
                response_text = llm_response.generations[0][0].text
            except (AttributeError, TypeError):
                # Fallback for LLMs that don't support agenerate_messages
                logger.warning("LLM does not support agenerate_messages, using fallback method")
                # Render full prompt with diagnosis
                rendered_msgs = self.chat_prompt.format_messages(
                    message=message,
                    emotion_data=emotion_data,
                    safety_data=safety_data,
                    diagnosis_data=diagnosis_data,
                    history=self._format_history(history)
                )
                prompt_text = "\n".join(m.content for m in rendered_msgs)
                # Use a synchronous approach as fallback
                sync_result = self.llm.generate([prompt_text])
                response_text = sync_result.generations[0][0].text
            
            # Attempt to update memory, but don't let it break response
            try:
                await self.memory.add("chat_history", [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response_text}
                ])
            except Exception as mem_error:
                logger.warning(f"Failed to update chat memory: {mem_error}")

            # Return the generated response
            return {
                "response": response_text,
                "timestamp": datetime.now().isoformat()
            }
        # Catch generation errors separately to still return fallback
        except Exception as e:
            logger.error(f"Error in chat response generation, returning fallback: {e}")
            return {"response": "I'm having trouble generating a response right now. Please try again later."}
    
    def _format_emotion_data(self, emotion_data: Dict[str, Any]) -> str:
        """Format emotional context for the prompt"""
        if not emotion_data:
            return "No emotional data available"
            
        return f"""Emotional State:
- Primary: {emotion_data.get('primary_emotion', 'unknown')}
- Secondary: {', '.join(emotion_data.get('secondary_emotions', []))}
- Intensity: {emotion_data.get('intensity', 'unknown')}
- Clinical Indicators: {', '.join(emotion_data.get('clinical_indicators', []))}"""
    
    def _format_safety_data(self, safety_data: Dict[str, Any]) -> str:
        """Format safety context for the prompt"""
        if not safety_data:
            return "No safety data available"
            
        return f"""Safety Assessment:
- Risk Level: {safety_data.get('risk_level', 'unknown')}
- Concerns: {', '.join(safety_data.get('concerns', []))}
- Warning Signs: {', '.join(safety_data.get('warning_signs', []))}
- Emergency Protocol: {'Yes' if safety_data.get('emergency_protocol') else 'No'}"""
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for the prompt"""
        if not history:
            return "No previous conversation history"
            
        formatted_history = []
        for i, msg in enumerate(history[-5:]):  # Limit to last 5 messages
            role = "User" if msg.get("role") == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg.get('content', '')}")
            
        return "\n".join(formatted_history)