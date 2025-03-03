from typing import Dict, Any, Optional, List
from langchain_anthropic import ChatAnthropic
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

class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        # Remove proxies argument if present
        kwargs.pop("proxies", None)
        super().__init__(*args, **kwargs)

class BaseAgent(Agent):
    def __init__(
        self,
        api_key: str,
        name: str = "base_agent",
        role: str = "A helpful AI assistant",
        description: str = "A base agent for mental health assistance",
        tools: Optional[List] = None,
        memory: Optional[Memory] = None,
        knowledge: Optional[AgentKnowledge] = None,
        show_tool_calls: bool = True,
        markdown: bool = True
    ):
        # Create a custom HTTP client for Anthropic
        http_client = CustomHTTPClient()
        
        # Initialize the ChatAnthropic model
        chat_model = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=api_key,
            temperature=0.7,
            max_tokens=2000,
        )
        
        # Create a default memory if none is provided
        if memory is None:
            # Create a langchain memory
            langchain_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            # Create a compatible memory for agno
            memory = Memory(
                memory_key="chat_history",
                chat_memory=langchain_memory,
                input_key="input",
                output_key="output",
                return_messages=True
            )
        
        super().__init__(
            name=name,
            role=role,
            model=chat_model,
            description=description,
            tools=tools or [],
            memory=memory,
            knowledge=knowledge or AgentKnowledge(),
            show_tool_calls=show_tool_calls,
            markdown=markdown
        )
        
        self.api_key = api_key
        
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
            response = await self.generate_response(
                query,
                context=full_context
            )
            
            # Update memory
            await self._update_memory(query, response)
            
            return {
                'response': response,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': self.name,
                    'confidence': self._calculate_confidence(response)
                }
            }
            
        except Exception as e:
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