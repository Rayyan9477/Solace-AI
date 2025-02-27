from typing import Dict, Any, Optional
from langchain_community.chat_models import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
import agno
from agno.agent import Agent
from agno.tools import tool as Tool  # Use the correct case or function name
from agno.memory import Memory
# Fix the Knowledge import - using the correct class from agno.knowledge
from agno.knowledge import VectorKnowledge  # Replace Knowledge with actual class
from datetime import datetime

class BaseAgent(Agent):
    def __init__(
        self,
        api_key: str,
        name: str,
        description: str,
        tools: Optional[list[Tool]] = None,
        memory: Optional[Memory] = None,
        knowledge: Optional[VectorKnowledge] = None  # Update type hint
    ):
        super().__init__(
            name=name,
            description=description,
            tools=tools or [],
            memory=memory,
            knowledge=knowledge
        )
        
        self.api_key = api_key
        self.llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=api_key,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Initialize Agno components
        self.memory = memory or agno.memory.ConversationMemory()
        self.knowledge = knowledge or agno.knowledge.VectorKnowledge()
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return response"""
        try:
            # Get context from memory and knowledge
            context = await self._get_context(input_data)
            
            # Process with tools
            tool_results = await self._process_with_tools(input_data, context)
            
            # Generate response
            response = await self._generate_response(input_data, context, tool_results)
            
            # Update memory
            await self._update_memory(input_data, response)
            
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
            
    async def _get_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant context from memory and knowledge"""
        memory_context = await self.memory.get(input_data.get('session_id'))
        knowledge_context = await self.knowledge.search(input_data.get('query', ''))
        
        return {
            'memory': memory_context,
            'knowledge': knowledge_context
        }
        
    async def _process_with_tools(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process input with available tools"""
        results = {}
        for tool in self.tools:
            if tool.should_use(input_data, context):
                results[tool.name] = await tool.run(input_data, context)
        return results
        
    async def _generate_response(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> str:
        """Generate response using LLM"""
        raise NotImplementedError("Subclasses must implement _generate_response")
        
    async def _update_memory(
        self,
        input_data: Dict[str, Any],
        response: Dict[str, Any]
    ) -> None:
        """Update agent memory"""
        await self.memory.add(
            session_id=input_data.get('session_id'),
            data={
                'input': input_data,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
        )
        
    def _calculate_confidence(self, response: Dict[str, Any]) -> float:
        """Calculate confidence score for the response"""
        # Implement confidence scoring logic
        return 0.8  # Default high confidence 