"""
Memory Factory - Centralized memory initialization for agents.

This module provides factory functions to create memory instances for agents,
eliminating the 250+ lines of duplicated memory initialization code across
multiple agent implementations.
"""

from typing import Optional, Dict, Any
from langchain.memory import ConversationBufferMemory
from agno.memory import Memory
import logging

logger = logging.getLogger(__name__)


def create_agent_memory(
    memory_key: str = "chat_history",
    input_key: str = "input",
    output_key: str = "output",
    return_messages: bool = True,
    storage_type: str = "local_storage",
    custom_config: Optional[Dict[str, Any]] = None
) -> Memory:
    """
    Create a standardized Memory instance for agents.

    This factory function centralizes memory initialization to ensure consistency
    across all agents and eliminate code duplication.

    Args:
        memory_key: Key for storing conversation history (default: "chat_history")
        input_key: Key for input messages (default: "input")
        output_key: Key for output messages (default: "output")
        return_messages: Whether to return messages as objects (default: True)
        storage_type: Type of storage to use (default: "local_storage")
        custom_config: Optional custom configuration to merge with defaults

    Returns:
        Memory: Configured Memory instance ready for use

    Example:
        >>> memory = create_agent_memory()
        >>> # Use in agent initialization
        >>> agent = MyAgent(model=llm, memory=memory)
    """
    try:
        # Create langchain memory instance
        langchain_memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages,
            input_key=input_key,
            output_key=output_key
        )

        # Create memory dict for agno Memory
        memory_dict = {
            "memory": "chat_memory",
            "storage": storage_type,
            "memory_key": memory_key,
            "chat_memory": langchain_memory,
            "input_key": input_key,
            "output_key": output_key,
            "return_messages": return_messages
        }

        # Merge custom configuration if provided
        if custom_config:
            memory_dict.update(custom_config)

        # Initialize and return Memory
        memory = Memory(**memory_dict)
        logger.debug(f"Created agent memory with key: {memory_key}")
        return memory

    except Exception as e:
        logger.error(f"Error creating agent memory: {str(e)}")
        raise


def create_stateless_memory() -> Memory:
    """
    Create a stateless memory instance for agents that don't need conversation history.

    Returns:
        Memory: Empty Memory instance
    """
    try:
        langchain_memory = ConversationBufferMemory(
            memory_key="temp_history",
            return_messages=False,
            input_key="input",
            output_key="output"
        )

        memory_dict = {
            "memory": "chat_memory",
            "storage": "temp_storage",
            "memory_key": "temp_history",
            "chat_memory": langchain_memory,
            "input_key": "input",
            "output_key": "output",
            "return_messages": False
        }

        return Memory(**memory_dict)

    except Exception as e:
        logger.error(f"Error creating stateless memory: {str(e)}")
        raise


def get_or_create_memory(
    provided_memory: Optional[Memory] = None,
    **factory_kwargs
) -> Memory:
    """
    Get provided memory or create a new one if None.

    This is the recommended function to use in agent __init__ methods.

    Args:
        provided_memory: Optional pre-existing memory instance
        **factory_kwargs: Arguments to pass to create_agent_memory() if creating new

    Returns:
        Memory: Either the provided memory or a newly created one

    Example:
        >>> memory = get_or_create_memory(provided_memory, memory_key="custom_history")
    """
    if provided_memory is not None:
        return provided_memory
    return create_agent_memory(**factory_kwargs)
