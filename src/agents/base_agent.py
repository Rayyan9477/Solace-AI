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
        """Process a query and return response"""
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
            
            # Update memory
            try:
                await self._update_memory(query, response)
            except Exception as e:
                logger.warning(f"Failed to update memory: {str(e)}")
            
            return {
                'response': response,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': self.name,
                    'confidence': self._calculate_confidence(response)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    async def _get_context(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get relevant context from memory and knowledge"""
        memory_context = await self.memory.get(context.get('session_id'))
        knowledge_context = await self.knowledge.search(query)
        
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
        await self.memory.add(
            data={
                'query': query,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
        )
        
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