from typing import Dict, Any, Optional, List
from langchain_core.messages import SystemMessage, HumanMessage
from agno.agent import Agent
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from datetime import datetime
from langchain.schema.language_model import BaseLanguageModel
import logging

# Import memory factory for centralized memory management (canonical location)
from src.memory.memory_factory import get_or_create_memory

logger = logging.getLogger(__name__)

# Import security validation (graceful fallback if not available)
try:
    from ..security import validate_user_message, ValidationSeverity
    SECURITY_VALIDATION_AVAILABLE = True
except ImportError:
    logger.warning("Security validation module not available")
    SECURITY_VALIDATION_AVAILABLE = False

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
        # Get or create memory using centralized factory
        memory = get_or_create_memory(memory)

        # Store the model for direct access
        self.llm = model

        # Store agent capabilities for orchestrator discovery
        self.capabilities = []  # Subclasses should override

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
        """
        Process a query and return response with supervision support.

        This method now includes input validation to protect against
        injection attacks and malicious input.
        """
        processing_start_time = datetime.now()

        try:
            # Validate input for security threats
            validation_result = None
            if SECURITY_VALIDATION_AVAILABLE:
                validation_result = validate_user_message(query)

                # Check validation severity
                if not validation_result.is_valid:
                    if validation_result.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]:
                        # Block critical/high severity threats
                        logger.error(
                            f"Security validation failed for query: {validation_result.errors}"
                        )
                        return {
                            'error': 'Input validation failed for security reasons',
                            'validation_errors': validation_result.errors,
                            'timestamp': datetime.now().isoformat(),
                            'metadata': {
                                'agent_name': self.name,
                                'security_blocked': True,
                                'processing_time': (datetime.now() - processing_start_time).total_seconds()
                            }
                        }
                    else:
                        # Log warnings but continue processing
                        logger.warning(
                            f"Input validation warnings: {validation_result.warnings}"
                        )

                # Use sanitized query
                query = validation_result.sanitized_value

                # Log validation warnings if any
                if validation_result.warnings:
                    logger.info(f"Input sanitized: {validation_result.warnings}")
            else:
                logger.warning("Security validation not available - processing without validation")
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
        """
        Synchronous fallback for generate_response when async is not supported.

        This method provides a robust fallback with proper context handling,
        error recovery, and consistent response formatting.

        Args:
            query: The user's query
            context: Optional context dictionary with memory, knowledge, etc.

        Returns:
            Dict with response text, metadata, and confidence score
        """
        processing_start = datetime.now()
        context = context or {}

        try:
            # Check if LLM supports synchronous generation
            if not hasattr(self.llm, 'generate'):
                logger.error(f"{self.name}: LLM does not support generate() method")
                return {
                    'response': "I apologize, but I'm unable to process your request at the moment due to a technical limitation.",
                    'error': 'LLM does not support synchronous generation',
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'agent_name': self.name,
                        'confidence': 0.0,
                        'processing_time': (datetime.now() - processing_start).total_seconds()
                    }
                }

            # Format query with context if available
            formatted_query = self._format_query_with_context(query, context)

            # Generate response
            llm_result = self.llm.generate([formatted_query])

            if not llm_result.generations or not llm_result.generations[0]:
                raise ValueError("LLM returned empty response")

            response_text = llm_result.generations[0][0].text

            # Calculate processing time and confidence
            processing_time = (datetime.now() - processing_start).total_seconds()
            confidence = self._calculate_confidence({'response': response_text})

            return {
                'response': response_text,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': self.name,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'fallback_method': 'sync_generation'
                }
            }

        except Exception as e:
            logger.error(f"Error in {self.name} sync response generation: {str(e)}", exc_info=True)
            processing_time = (datetime.now() - processing_start).total_seconds()

            return {
                'response': (
                    "I apologize, but I encountered an error while processing your request. "
                    "Please try rephrasing your question or contact support if the issue persists."
                ),
                'error': str(e),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': self.name,
                    'confidence': 0.1,
                    'processing_time': processing_time,
                    'fallback_method': 'error_recovery'
                }
            }

    def _format_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """
        Format user query with contextual information for enhanced LLM understanding.

        Enriches the query with conversation history, relevant knowledge, and emotional
        state to provide the LLM with comprehensive context for generating more accurate
        and empathetic responses.

        Args:
            query (str): User's raw query or message
            context (Dict[str, Any]): Context dictionary containing:
                - 'memory' (list[dict], optional): Conversation history items, each with:
                    - 'role' (str): Message source ('user', 'assistant', 'system')
                    - 'content' (str): Message content
                    - 'timestamp' (str, optional): ISO format timestamp
                - 'knowledge' (list[dict], optional): Relevant knowledge items, each with:
                    - 'content' (str): Knowledge text
                    - 'relevance_score' (float): Similarity score (0-1)
                    - 'source' (str): Knowledge source identifier
                - 'emotion' (dict, optional): Detected emotional state with:
                    - 'primary_emotion' (str): Main emotion label
                    - 'intensity' (float): Emotion strength (1-10)
                    - 'confidence' (float): Detection confidence (0-1)

        Returns:
            str: Formatted prompt string combining query, context, and agent role

        Example:
            >>> context = {
            ...     'memory': [
            ...         {'role': 'user', 'content': 'I have been feeling stressed'},
            ...         {'role': 'assistant', 'content': 'I understand. Let\\'s explore that.'}
            ...     ],
            ...     'knowledge': [
            ...         {'content': 'Stress management techniques include...', 'relevance_score': 0.85}
            ...     ],
            ...     'emotion': {'primary_emotion': 'anxious', 'intensity': 7.2}
            ... }
            >>> formatted = self._format_query_with_context("What can I do about it?", context)
            >>> print(formatted)
            User Query: What can I do about it?

            Recent Context:
            - user: I have been feeling stressed
            - assistant: I understand. Let's explore that.

            Relevant Knowledge:
            - Stress management techniques include...

            User Emotion: anxious

            As Mental Health Assistant, provide a helpful response:

        Note:
            - Memory is limited to last 3 items to prevent context overload
            - Knowledge is limited to top 2 most relevant items
            - Content is truncated to 100 characters for conciseness
        """
        formatted_parts = [f"User Query: {query}"]

        # Add memory context if available
        if 'memory' in context and context['memory']:
            memory_items = context['memory'][-3:]  # Last 3 memory items
            if memory_items:
                formatted_parts.append("\nRecent Context:")
                for item in memory_items:
                    if isinstance(item, dict):
                        role = item.get('role', 'unknown')
                        content = item.get('content', '')
                        formatted_parts.append(f"- {role}: {content[:100]}...")

        # Add knowledge context if available
        if 'knowledge' in context and context['knowledge']:
            knowledge_items = context['knowledge'][:2]  # Top 2 knowledge items
            if knowledge_items:
                formatted_parts.append("\nRelevant Knowledge:")
                for item in knowledge_items:
                    if isinstance(item, dict):
                        formatted_parts.append(f"- {item.get('content', '')[:100]}...")

        # Add any additional context hints
        if context.get('emotion'):
            formatted_parts.append(f"\nUser Emotion: {context['emotion'].get('primary_emotion', 'unknown')}")

        formatted_parts.append(f"\nAs {self.role}, provide a helpful response:")

        return "\n".join(formatted_parts)

    def get_capabilities(self) -> List[str]:
        """
        Return list of capabilities this agent provides.
        Override in subclasses to specify specific capabilities.

        Returns:
            List of capability identifiers (e.g., ["chat", "empathy", "crisis_detection"])
        """
        return self.capabilities if self.capabilities else [self.name]