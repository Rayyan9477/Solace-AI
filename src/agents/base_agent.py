from typing import Dict, Any, Optional, List
from langchain_anthropic.chat_models import ChatAnthropicMessages
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
import agno
from agno.agent import Agent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from datetime import datetime
import anthropic
import httpx
from langchain.schema.language_model import BaseLanguageModel
import logging

logger = logging.getLogger(__name__)

class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        # Remove proxies argument if present
        kwargs.pop("proxies", None)
        super().__init__(*args, **kwargs)

class BaseAgent(Agent):
    def __init__(
        self,
        model: BaseLanguageModel,
        name: str = "base_agent",
        role: str = "A helpful AI assistant",
        description: str = "A base agent for mental health assistance",
        tools: Optional[List] = None,
        memory: Optional[Memory] = None,
        knowledge: Optional[AgentKnowledge] = None,
        show_tool_calls: bool = True,
        markdown: bool = True
    ):
        # Create a default memory if none is provided
        if memory is None:
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
        
        # Store the model for direct access
        self.llm = model
        
        super().__init__(
            name=name,
            role=role,
            model=model,
            description=description,
            tools=tools or [],
            memory=memory,
            knowledge=knowledge or AgentKnowledge(),
            show_tool_calls=show_tool_calls,
            markdown=markdown
        )
        
    async def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a query and return response with supervision support"""
        processing_start_time = datetime.now()
        
        try:
            # Get context from memory and knowledge
            full_context = await self._get_context(query, context or {})
            
            # Generate response using agent framework
            try:
                response = await self.generate_response(
                    query,
                    context=full_context
                )
            except (AttributeError, TypeError) as e:
                # Fallback for LLMs that don't support async generation
                logger.warning(f"Async generation not supported: {str(e)}, using fallback method")
                response = self.generate_response_sync(
                    query,
                    context=full_context
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - processing_start_time).total_seconds()
            
            # Prepare result with supervision metadata
            result = {
                'response': response,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': self.name,
                    'confidence': self._calculate_confidence(response),
                    'processing_time': processing_time
                }
            }
            
            # Add context updates for supervisor tracking
            context_updates = self._generate_context_updates(query, response, full_context)
            if context_updates:
                result['context_updates'] = context_updates
            
            # Store result in central vector DB if it has metadata related to assessments
            try:
                if hasattr(self, 'store_to_vector_db') and callable(self.store_to_vector_db):
                    await self.store_to_vector_db(query, response, context)
            except Exception as store_error:
                logger.error(f"Error storing data in vector DB: {str(store_error)}")
            
            # Update memory
            try:
                await self._update_memory(query, response)
            except Exception as e:
                logger.warning(f"Failed to update memory: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'agent_name': self.name,
                    'processing_time': (datetime.now() - processing_start_time).total_seconds()
                }
            }
            
    async def _get_context(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get relevant context from memory and knowledge"""
        try:
            memory_context = await self.memory.get("chat_history", [])
        except Exception as e:
            logger.warning(f"Failed to get memory context: {str(e)}")
            memory_context = []
            
        try:
            knowledge_context = await self.knowledge.search(query)
        except Exception as e:
            logger.warning(f"Failed to get knowledge context: {str(e)}")
            knowledge_context = []
        
        return {
            'memory': memory_context,
            'knowledge': knowledge_context,
            **context
        }
        
    async def _update_memory(
        self,
        query: str,
        response: Dict[str, Any]
    ) -> None:
        """Update agent memory"""
        try:
            if hasattr(self.memory, 'chat_memory'):
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response.get('response', ''))
            else:
                logger.warning("Memory does not have chat_memory attribute")
        except Exception as e:
            logger.warning(f"Failed to update memory: {str(e)}")
        
    def _calculate_confidence(self, response: Dict[str, Any]) -> float:
        """Calculate confidence score for the response"""
        # Base confidence calculation
        confidence = 0.8
        
        # Lower confidence if response is empty or has errors
        if not response or response.get('error'):
            confidence *= 0.5
            
        # Lower confidence if response is too short
        if isinstance(response.get('response'), str) and len(response['response']) < 50:
            confidence *= 0.7
            
        return confidence
    
    def _generate_context_updates(self, query: str, response: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate context updates for supervisor tracking"""
        context_updates = {}
        
        # Add agent-specific context information
        context_updates[f"{self.name}_last_interaction"] = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(str(query)),
            "response_length": len(str(response)) if response else 0,
            "confidence": self._calculate_confidence(response) if isinstance(response, dict) else 0.5
        }
        
        # Add any agent-specific context updates
        if hasattr(self, '_get_agent_specific_context'):
            agent_context = self._get_agent_specific_context(query, response, context)
            if agent_context:
                context_updates.update(agent_context)
        
        return context_updates 

    def generate_response_sync(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous version of generate_response for fallback"""
        try:
            # Simple implementation for fallback
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate([query])
                return response.generations[0][0].text
            else:
                return "I'm having trouble generating a response right now."
        except Exception as e:
            logger.error(f"Error in sync response generation: {str(e)}")
            return "I'm having trouble generating a response right now."