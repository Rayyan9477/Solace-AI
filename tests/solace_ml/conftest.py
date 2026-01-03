"""Pytest fixtures for solace_ml tests."""
from __future__ import annotations
import pytest
from solace_ml.llm_client import (
    LLMSettings, Message, MessageRole, ToolDefinition, ToolParameter,
)
from solace_ml.anthropic import AnthropicSettings
from solace_ml.openai import OpenAISettings
from solace_ml.embeddings import EmbeddingSettings


@pytest.fixture
def llm_settings():
    """Default LLM settings for testing."""
    return LLMSettings(
        max_tokens=1024,
        temperature=0.5,
        max_retries=2
    )


@pytest.fixture
def anthropic_settings():
    """Anthropic settings for testing."""
    return AnthropicSettings(
        model="claude-sonnet-4-20250514",
        max_tokens=1024
    )


@pytest.fixture
def openai_settings():
    """OpenAI settings for testing."""
    return OpenAISettings(
        model="gpt-4o-mini",
        max_tokens=1024
    )


@pytest.fixture
def embedding_settings():
    """Embedding settings for testing."""
    return EmbeddingSettings(
        batch_size=10,
        cache_enabled=True
    )


@pytest.fixture
def sample_messages():
    """Sample message list for testing."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello!"),
        Message(role=MessageRole.ASSISTANT, content="Hi there! How can I help?"),
        Message(role=MessageRole.USER, content="What is AI?"),
    ]


@pytest.fixture
def sample_tool():
    """Sample tool definition for testing."""
    return ToolDefinition(
        name="get_weather",
        description="Get current weather for a city",
        parameters=[
            ToolParameter(name="city", type="string", description="City name"),
            ToolParameter(name="unit", type="string", description="Temperature unit",
                         enum=["celsius", "fahrenheit"], required=False)
        ]
    )


@pytest.fixture
def sample_tools():
    """Multiple sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="get_weather",
            description="Get current weather",
            parameters=[ToolParameter(name="city", type="string", description="City")]
        ),
        ToolDefinition(
            name="search_web",
            description="Search the web",
            parameters=[ToolParameter(name="query", type="string", description="Search query")]
        ),
        ToolDefinition(
            name="calculate",
            description="Perform calculation",
            parameters=[
                ToolParameter(name="expression", type="string", description="Math expression")
            ]
        )
    ]
