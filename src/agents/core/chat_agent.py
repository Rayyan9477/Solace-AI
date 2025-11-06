from typing import Optional, Dict, Any, List
import os
# Import Gemini integration
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.memory import ConversationBufferMemory
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from ..base.base_agent import BaseAgent
import logging
from datetime import datetime
import json
import numpy as np
from collections import deque
from typing import Deque
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use provider-agnostic factory
from src.models.llm import get_llm
# Import for therapy integration
from src.agents.clinical.therapy_agent import TherapyAgent
# Import the enhanced memory components
from src.utils.context_aware_memory import ContextAwareMemoryAdapter
from src.memory.semantic_memory import SemanticMemoryManager
# Import vector database integration
from src.utils.vector_db_integration import add_user_data, get_user_data

logger = logging.getLogger(__name__)

class ChatAgent(BaseAgent):
    def __init__(self, 
                 model: BaseLanguageModel = None,
                 user_id: str = "default_user",
                 enable_semantic_memory: bool = True,
                 conversation_summary_threshold: int = 15,
                 memory_storage_dir: Optional[str] = None):
        # If no model is provided, create model via provider-agnostic factory
        if model is None:
            from src.config.settings import AppConfig
            provider_config = {
                "provider": AppConfig.LLM_CONFIG.get("provider", "gemini"),
                "model_name": AppConfig.LLM_CONFIG.get("model", AppConfig.MODEL_NAME),
                "api_key": AppConfig.LLM_CONFIG.get("api_key", os.environ.get("GEMINI_API_KEY", "")),
                "temperature": AppConfig.LLM_CONFIG.get("temperature", 0.7),
                "max_output_tokens": 4096,
                "top_p": AppConfig.LLM_CONFIG.get("top_p", 0.95),
            }
            model = get_llm(provider_config)
            logger.info(f"Created default LLM for ChatAgent provider={provider_config['provider']}")

        # Create a langchain memory instance
        langchain_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )

        # Create memory dict for agno Memory
        memory_dict = {
            "memory": "chat_memory",
            "storage": "local_storage",
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
            description="Supportive mental health chat assistant with Gemini AI",
            tools=[],
            memory=memory,
            knowledge=AgentKnowledge()
        )
        
        # Create therapy agent for accessing therapeutic techniques
        self.therapy_agent = None
        
        # Initialize enhanced memory components if enabled
        self.user_id = user_id
        self.enable_semantic_memory = enable_semantic_memory
        self.context_memory = None
        if enable_semantic_memory:
            try:
                self.context_memory = ContextAwareMemoryAdapter(
                    user_id=user_id,
                    conversation_threshold=conversation_summary_threshold,
                    storage_dir=memory_storage_dir
                )
                logger.info(f"Initialized enhanced context-aware memory for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to initialize context-aware memory: {str(e)}")
                self.enable_semantic_memory = False
        
        # Personality adaptation state
        self.personality_adaptations = {
            "current_tone": "supportive",
            "warmth": 0.7,
            "formality": 0.4,
            "complexity": 0.5,
            "emotion_mirroring": 0.6
        }
        
        # Conversation metrics
        self.conversation_metrics = {
            "total_exchanges": 0,
            "last_activity_time": datetime.now(),
            "emotion_trend": [],
            "topic_shifts": []
        }

        # Enhanced chat prompt with advanced context and personality adaptations
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a supportive mental health assistant powered by Google Gemini 2.0 AI.
Your role is to provide empathetic, culturally sensitive, and non-judgmental support while:
1. Creating a safe, empathetic space through your responses
2. Validating emotions and experiences with genuine understanding
3. Offering evidence-based coping strategies appropriate to the user's situation
4. Encouraging professional help when appropriate without being pushy
5. Avoiding diagnostic statements or medical advice
6. Personalizing your approach based on the user's personality profile and cultural background
7. Responding with cultural sensitivity and awareness

Guidelines for Empathetic Responses:
- Mirror the user's emotional state appropriately in your tone
- Validate their feelings and experiences without reinforcing negative patterns
- Use a warm, conversational style that feels human and authentic
- Suggest practical and culturally relevant coping strategies
- Adapt your responses based on personality assessment data
- Maintain appropriate professional boundaries while being genuinely supportive
- Recognize cultural nuances in emotional expression and mental health perspectives
- Ensure responses are sensitive to different cultural backgrounds and values

{personality_adaptations}

In every response:
1. Show that you understand the user's emotions
2. Validate their experiences
3. Provide helpful perspective or coping strategies
4. End with a supportive statement or gentle question

When therapeutic techniques are included, integrate them naturally into your response, 
emphasizing how these evidence-based strategies can help with their specific situation."""),
            HumanMessage(content="""User Message: {message}
Emotional Context: {emotion_data}
Safety Context: {safety_data}
Diagnosis Context: {diagnosis_data}
Personality Profile: {personality_data}
Previous Conversation: {history}
Therapeutic Techniques: {therapeutic_data}
Memory Context: {memory_context}

Provide a supportive, empathetic response that addresses the user's emotional needs and current context while being culturally sensitive. Tailor your response to their personality profile and emotional state. When therapeutic techniques are provided, integrate them naturally as actionable steps.""")
        ])

    async def generate_response(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a supportive response to a user message

        Args:
            message: The user's message
            context: Additional context including emotion, safety, and personality data

        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Initialize context if not provided
            if context is None:
                context = {}
            
            # Get conversation history
            history = await self.memory.get("chat_history", [])
            
            # Update conversation metrics
            self._update_conversation_metrics(message, context)
            
            # Process enhanced memory and context if enabled
            memory_context = "No additional memory context available"
            if self.enable_semantic_memory and self.context_memory:
                try:
                    # Add message to context memory
                    await self.context_memory.add_message(message, role="user", metadata={"timestamp": datetime.now().isoformat()})
                    
                    # Update emotion context if available
                    if "emotion" in context:
                        await self.context_memory.update_emotion_context(context["emotion"])
                    
                    # Update personality context if available
                    if "personality" in context:
                        await self.context_memory.update_personality_context(context["personality"])
                    
                    # Get relevant context for current message
                    relevant_context = await self.context_memory.get_relevant_context(message)
                    
                    # Get personality adjustments based on context
                    personality_adjustments = self.context_memory.get_current_personality_adjustments()
                    self.personality_adaptations.update(personality_adjustments)
                    
                    # Format context for prompt
                    memory_context = self.context_memory.format_context_for_prompt()
                except Exception as e:
                    logger.error(f"Error processing enhanced memory: {str(e)}")
                    memory_context = f"Memory processing error: {str(e)}"
            
            # Format personality adaptations for the prompt
            personality_adaptations_str = self._format_personality_adaptations()
            
            # Format context data
            emotion_data = self._format_emotion_data(context.get("emotion", {}))
            safety_data = self._format_safety_data(context.get("safety", {}))
            diagnosis_data = context.get("diagnosis", "No diagnosis available")
            personality_data = self._format_personality_data(context.get("personality", {}))
            
            # Get therapeutic techniques if available in context
            therapeutic_data = "No therapeutic techniques available"
            if "therapeutic_techniques" in context:
                therapeutic_data = context.get("therapeutic_techniques", {}).get("formatted_techniques", "")

            # Generate response
            try:
                # Unified wrapper exposes agenerate for list of prompts; for messages we convert to prompt
                rendered_msgs = self.chat_prompt.format_messages(
                    message=message,
                    emotion_data=emotion_data,
                    safety_data=safety_data,
                    diagnosis_data=diagnosis_data,
                    personality_data=personality_data,
                    therapeutic_data=therapeutic_data,
                    history=self._format_history(history),
                    memory_context=memory_context,
                    personality_adaptations=personality_adaptations_str
                )
                prompt_text = "\n".join(m.content for m in rendered_msgs)
                llm_result = await self.llm._agenerate([prompt_text])  # returns LLMResult
                response_text = llm_result.generations[0][0].text
            except (AttributeError, TypeError):
                # Fallback for LLMs that don't support agenerate_messages
                logger.warning("LLM does not support agenerate_messages, using fallback method")
                # Render full prompt with context
                rendered_msgs = self.chat_prompt.format_messages(
                    message=message,
                    emotion_data=emotion_data,
                    safety_data=safety_data,
                    diagnosis_data=diagnosis_data,
                    personality_data=personality_data,
                    therapeutic_data=therapeutic_data,
                    history=self._format_history(history),
                    memory_context=memory_context,
                    personality_adaptations=personality_adaptations_str
                )
                prompt_text = "\n".join(m.content for m in rendered_msgs)
                # Use a synchronous approach as fallback
                sync_result = self.llm.generate([prompt_text])
                response_text = sync_result.generations[0][0].text

            # Check if we need to enhance the response with therapeutic techniques
            if "workflow_id" in context and context["workflow_id"] == "therapeutic_chat":
                # Check if therapy agent exists, create if not
                if self.therapy_agent is None:
                    logger.info("Creating therapy agent for therapeutic chat workflow")
                    self.therapy_agent = TherapyAgent(model_provider=self.llm)
                
                # Process the message with therapy agent
                if "therapeutic_techniques" not in context:
                    therapy_result = await self.therapy_agent.process(message, context)
                    
                    # Enhance response with therapeutic techniques if available
                    if therapy_result and therapy_result.get("formatted_techniques"):
                        response_text = self.therapy_agent.enhance_response(response_text, therapy_result)

            # Update original memory
            try:
                await self.memory.add("chat_history", [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response_text}
                ])
            except Exception as mem_error:
                logger.warning(f"Failed to update chat memory: {str(mem_error)}")
            
            # Update semantic memory if enabled
            if self.enable_semantic_memory and self.context_memory:
                try:
                    await self.context_memory.add_message(
                        message=response_text,
                        role="assistant",
                        metadata={"timestamp": datetime.now().isoformat()}
                    )
                except Exception as e:
                    logger.error(f"Failed to update semantic memory with response: {str(e)}")

            # Store chat messages in the central vector database
            await self.store_to_vector_db(message, {"response": response_text}, context)

            # Return the generated response with metadata
            return {
                "response": response_text,
                "timestamp": datetime.now().isoformat(),
                "personality_tone": self.personality_adaptations.get("current_tone", "supportive")
            }
        # Catch generation errors separately to still return fallback
        except Exception as e:
            logger.error(f"Error in chat response generation, returning fallback: {e}")
            return {"response": "I'm having trouble generating a response right now. Please try again later."}
    
    async def store_to_vector_db(self, query: str, response: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Store chat messages in the central vector database
        
        Args:
            query: User's query
            response: Agent's response
            context: Processing context
        """
        try:
            # Check if we have a valid response
            if isinstance(response, dict) and 'response' in response:
                chat_data = {
                    "user_message": query,
                    "assistant_response": response.get('response', ''),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add emotion data if available in the context
                if context and 'emotion_analysis' in context:
                    chat_data['emotion_data'] = context['emotion_analysis']
                    
                # Add personality data if available in the context
                if context and 'personality' in context:
                    chat_data['personality_context'] = context['personality']
                
                # Add user ID if available in context
                if context and "user_id" in context:
                    chat_data["user_id"] = context["user_id"]
                
                # Store in vector database
                doc_id = add_user_data("conversation", chat_data)
                
                if doc_id:
                    logger.info(f"Stored chat message in vector DB: {doc_id}")
                else:
                    logger.warning("Failed to store chat message in vector DB")
            
        except Exception as e:
            logger.error(f"Error storing chat data in vector DB: {str(e)}")
    
    def _update_conversation_metrics(self, message: str, context: Dict[str, Any]) -> None:
        """Update conversation metrics based on current exchange"""
        try:
            # Update total exchanges
            self.conversation_metrics["total_exchanges"] += 1
            
            # Update last activity time
            self.conversation_metrics["last_activity_time"] = datetime.now()
            
            # Track emotion if available
            if context and "emotion" in context and "primary_emotion" in context["emotion"]:
                # Extract current emotion
                current_emotion = context["emotion"]["primary_emotion"]
                
                # Add to emotion trend
                self.conversation_metrics["emotion_trend"].append({
                    "emotion": current_emotion,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Limit emotion trend history
                if len(self.conversation_metrics["emotion_trend"]) > 10:
                    self.conversation_metrics["emotion_trend"] = self.conversation_metrics["emotion_trend"][-10:]
        except Exception as e:
            logger.error(f"Error updating conversation metrics: {str(e)}")

    def _format_personality_adaptations(self) -> str:
        """Format personality adaptations for prompt inclusion"""
        try:
            tone = self.personality_adaptations.get("current_tone", "supportive")
            warmth = self.personality_adaptations.get("warmth", 0.7)
            formality = self.personality_adaptations.get("formality", 0.4)
            complexity = self.personality_adaptations.get("complex", 0.5)
            
            # Convert values to descriptive terms
            warmth_desc = "very warm" if warmth > 0.8 else "warm" if warmth > 0.6 else "moderately warm" if warmth > 0.4 else "somewhat reserved"
            formality_desc = "formality" if formality > 0.7 else "semi-formal" if formality > 0.5 else "conversational" if formality > 0.3 else "casual"
            complexity_desc = "detailed" if complexity > 0.7 else "balanced" if complexity > 0.4 else "simple and direct"
            
            # Form the adaptation instruction
            return f"""Current personality adaptation:
- Communication tone: {tone}
- Style: {warmth_desc}, {formality_desc}, and {complexity_desc}
- The response should reflect these personality characteristics while remaining authentic and empathetic."""
            
        except Exception as e:
            logger.error(f"Error formatting personality adaptations: {str(e)}")
            return "Maintain a warm, empathetic, and supportive tone in your response."

    def _format_emotion_data(self, emotion_data: Dict[str, Any]) -> str:
        """Format emotional context for the prompt"""
        if not emotion_data:
            return "No emotional data available"

        # Check if we have voice emotion data
        voice_emotion = emotion_data.get('voice_emotion', {})
        voice_primary = voice_emotion.get('primary_emotion', 'unknown') if voice_emotion else 'unknown'
        voice_confidence = voice_emotion.get('confidence', 0.0) if voice_emotion else 0.0
        
        # Check if we have voice emotions detailed data
        voice_emotions_detailed = emotion_data.get('voice_emotions_detailed', {})
        
        # Format standard emotion data
        standard_emotion = f"""Emotional State (Text Analysis):
- Primary: {emotion_data.get('primary_emotion', 'unknown')}
- Secondary: {', '.join(emotion_data.get('secondary_emotions', []))}
- Intensity: {emotion_data.get('intensity', 'unknown')}
- Clinical Indicators: {', '.join(emotion_data.get('clinical_indicators', []))}"""

        # If we have voice emotion data with decent confidence, include it
        if voice_emotion and voice_confidence > 0.4:
            # Get top 3 voice emotions for display
            top_voice_emotions = []
            if voice_emotions_detailed:
                sorted_emotions = sorted(voice_emotions_detailed.items(), key=lambda x: x[1], reverse=True)[:3]
                top_voice_emotions = [f"{emotion}: {score:.2f}" for emotion, score in sorted_emotions]
            
            voice_emotion_str = f"""

Voice Emotion Analysis:
- Primary: {voice_primary}
- Confidence: {voice_confidence:.2f}
- Top emotions: {', '.join(top_voice_emotions) if top_voice_emotions else 'None detected'}
- Congruence: {emotion_data.get('congruence', 'unknown')}"""
            
            return standard_emotion + voice_emotion_str
        
        return standard_emotion

    def _format_safety_data(self, safety_data: Dict[str, Any]) -> str:
        """Format safety context for the prompt"""
        if not safety_data:
            return "No safety data available"

        return f"""Safety Assessment:
- Risk Level: {safety_data.get('risk_level', 'unknown')}
- Concerns: {', '.join(safety_data.get('concerns', []))}
- Warning Signs: {', '.join(safety_data.get('warning_signs', []))}
- Emergency Protocol: {'Yes' if safety_data.get('emergency_protocol') else 'No'}"""

    def _format_personality_data(self, personality_data: Dict[str, Any]) -> str:
        """Format personality context for the prompt"""
        if not personality_data:
            return "No personality data available"

        try:
            assessment_type = personality_data.get("type", "unknown")
            results = personality_data.get("results", {})

            if assessment_type == "big_five":
                # Format Big Five results
                traits = results.get("traits", {})
                if not traits:
                    return "Big Five assessment completed, but no detailed results available."

                trait_summaries = []
                for trait_name, trait_data in traits.items():
                    score = trait_data.get("score", 50)
                    category = trait_data.get("category", "average")
                    trait_summaries.append(f"{trait_name.capitalize()}: {category} ({int(score)}%)")

                return f"""Big Five (OCEAN) Profile:
- {trait_summaries[0] if len(trait_summaries) > 0 else 'No data'}
- {trait_summaries[1] if len(trait_summaries) > 1 else 'No data'}
- {trait_summaries[2] if len(trait_summaries) > 2 else 'No data'}
- {trait_summaries[3] if len(trait_summaries) > 3 else 'No data'}
- {trait_summaries[4] if len(trait_summaries) > 4 else 'No data'}"""

            elif assessment_type == "mbti":
                # Format MBTI results
                personality_type = results.get("type", "")
                type_name = results.get("type_name", "")

                if not personality_type:
                    return "MBTI assessment completed, but no type determined."

                # Get key strengths and challenges
                strengths = results.get("strengths", [])
                weaknesses = results.get("weaknesses", [])

                strengths_text = "\n  - " + "\n  - ".join(strengths[:3]) if strengths else "None specified"
                challenges_text = "\n  - " + "\n  - ".join(weaknesses[:3]) if weaknesses else "None specified"

                return f"""MBTI Profile:
- Type: {personality_type} ({type_name})
- Key Strengths:{strengths_text}
- Potential Challenges:{challenges_text}"""

            else:
                return f"Personality assessment of type '{assessment_type}' completed, but format is not recognized."

        except Exception as e:
            logger.error(f"Error formatting personality data: {str(e)}")
            return "Personality assessment completed, but there was an error processing the results."

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for the prompt"""
        if not history:
            return "No previous conversation history"

        formatted_history = []
        for i, msg in enumerate(history[-5:]):  # Limit to last 5 messages
            role = "User" if msg.get("role") == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg.get('content', '')}")

        return "\n".join(formatted_history)
    
    async def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the current conversation
        
        Returns:
            Dictionary containing the summary and metadata
        """
        if self.enable_semantic_memory and self.context_memory:
            try:
                return await self.context_memory.generate_summary()
            except Exception as e:
                logger.error(f"Error generating conversation summary: {str(e)}")
                
        # Fallback summary if semantic memory is not available
        try:
            history = await self.memory.get("chat_history", [])
            
            if not history:
                return {
                    "summary_text": "No conversation to summarize.",
                    "message_count": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in history[-10:]  # Use last 10 messages
            ])
            
            return {
                "summary_text": conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text,
                "message_count": len(history),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating fallback summary: {str(e)}")
            return {
                "summary_text": f"Error generating summary: {str(e)}",
                "message_count": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def search_conversation_history(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search conversation history using semantic search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching conversation entries
        """
        if self.enable_semantic_memory and self.context_memory:
            try:
                return await self.context_memory.search_conversations(query, top_k)
            except Exception as e:
                logger.error(f"Error searching conversation history: {str(e)}")
        
        # Return empty list if semantic memory is not available or if search fails
        return []