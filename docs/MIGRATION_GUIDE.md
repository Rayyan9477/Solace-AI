# 🔄 Migration Guide: New Modular Architecture

## Overview

This guide helps you migrate from the old monolithic architecture to the new modular, pluggable system. The new architecture provides better maintainability, testability, and extensibility.

## 🎯 Key Benefits of Migration

- **Easy Provider Switching**: Change LLM providers with configuration only
- **Better Testing**: Dependency injection enables easy mocking
- **Loose Coupling**: Interfaces separate contracts from implementations  
- **Event-Driven**: Components communicate through events
- **Configuration Management**: Centralized, validated configuration
- **Error Handling**: Structured exceptions with context

## 📋 Migration Checklist

- [ ] Update imports to use new interfaces
- [ ] Replace direct instantiation with factories
- [ ] Configure dependency injection
- [ ] Update configuration format
- [ ] Migrate to event-driven communication
- [ ] Update error handling
- [ ] Add tests with mocked dependencies

## 🔄 Step-by-Step Migration

### 1. LLM Provider Migration

#### ❌ Old Approach
```python
# Direct instantiation - hard to change
from src.models.gemini_llm import GeminiLLM

class ChatAgent:
    def __init__(self):
        self.llm = GeminiLLM(api_key="sk-...")  # Hardcoded
        
    async def process(self, message):
        response = self.llm.generate_content(message)
        return response.text
```

#### ✅ New Approach
```python
# Factory-based - easy to switch providers
from src.core.factories.llm_factory import LLMFactory
from src.core.interfaces.llm_interface import LLMInterface, Message, MessageRole

class ChatAgent:
    def __init__(self, llm_provider: LLMInterface):
        self.llm_provider = llm_provider  # Injected
        
    async def process(self, message: str):
        messages = [Message(role=MessageRole.USER, content=message)]
        response = await self.llm_provider.generate_response(messages)
        return response.content
```

#### Configuration-Based Provider Selection
```python
# config.json
{
    "llm": {
        "provider": "gemini",  # Change to "openai" to switch
        "model_name": "gemini-2.0-flash",
        "api_key": "${GEMINI_API_KEY}",
        "temperature": 0.7,
        "max_tokens": 2000
    }
}

# Application setup
from src.infrastructure.config.config_manager import ConfigManager

config_manager = ConfigManager()
await config_manager.load_all()

llm_config = config_manager.get_section("llm")
llm_provider = await LLMFactory.create_from_config(llm_config)
```

### 2. Dependency Injection Migration

#### ❌ Old Approach
```python
# Manual dependency management
class ChatAgent:
    def __init__(self):
        self.llm = GeminiLLM()
        self.vector_store = VectorStore()
        self.memory = ConversationMemory()
        self.personality = ChatbotPersonality()
        # Hard to test, tight coupling
```

#### ✅ New Approach
```python
# Dependency injection
from src.infrastructure.di.decorators import service
from src.core.interfaces import LLMInterface, StorageInterface

@service(ChatAgent, lifecycle="singleton")
class ChatAgent:
    def __init__(self, 
                 llm: LLMInterface,
                 storage: StorageInterface,
                 event_bus: EventBus):
        self.llm = llm          # Injected - easy to mock
        self.storage = storage   # Injected - easy to test
        self.event_bus = event_bus
```

#### Container Setup
```python
from src.infrastructure.di.container import DIContainer

async def setup_container():
    container = DIContainer()
    
    # Register LLM provider
    llm_provider = await LLMFactory.create_from_config(llm_config)
    container.register_instance(LLMInterface, llm_provider)
    
    # Register storage
    container.register_singleton(StorageInterface, VectorStorageProvider)
    
    # Register agents (auto-wired)
    container.register_singleton(ChatAgent)
    
    return container
```

### 3. Configuration Migration

#### ❌ Old Approach
```python
# Hardcoded in multiple files
class AppConfig:
    GEMINI_API_KEY = "hardcoded-key"
    MODEL_NAME = "gemini-2.0-flash"
    TEMPERATURE = 0.7
    DEBUG = True
```

#### ✅ New Approach
```python
# config/app.json
{
    "app": {
        "name": "Contextual-Chatbot",
        "version": "2.0.0",
        "debug": false,
        "environment": "production"
    },
    "llm": {
        "provider": "gemini",
        "model_name": "gemini-2.0-flash",
        "api_key": "${GEMINI_API_KEY}",
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "storage": {
        "type": "vector",
        "provider": "faiss",
        "storage_path": "./data/vectors"
    }
}

# Usage with validation
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.config.schema import ConfigSchema, ConfigField

# Define schema for validation
llm_schema = ConfigSchema("llm", fields=[
    ConfigField("provider", str, required=True, 
                choices=["gemini", "openai", "anthropic"]),
    ConfigField("model_name", str, required=True),
    ConfigField("temperature", float, default=0.7, 
                min_value=0.0, max_value=2.0),
    ConfigField("max_tokens", int, default=2000, 
                min_value=1, max_value=8192)
])

config = ConfigManager()
config.register_schema("llm", llm_schema)

# Load and validate
await config.load_all()
validation = config.validate_schema("llm")
if not validation.is_valid:
    print(f"Config errors: {validation.errors}")
```

### 4. Event-Driven Communication Migration

#### ❌ Old Approach
```python
# Direct method calls - tight coupling
class ChatAgent:
    def __init__(self):
        self.emotion_agent = EmotionAgent()
        self.safety_agent = SafetyAgent()
        
    async def process(self, message):
        # Direct calls
        emotion = self.emotion_agent.analyze(message)
        safety_check = self.safety_agent.check(message)
        
        if not safety_check.is_safe:
            return safety_check.response
            
        # Process with emotion context
        return self.generate_response(message, emotion)
```

#### ✅ New Approach
```python
# Event-driven - loose coupling
from src.core.interfaces.event_interface import EventBus, Event, SystemEvents

class ChatAgent:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        
    async def process(self, request: AgentRequest):
        # Publish event
        await self.event_bus.publish(Event(
            event_type=SystemEvents.AGENT_REQUEST_RECEIVED,
            payload={
                "agent_id": self.agent_id,
                "request_id": request.request_id,
                "content": request.content
            }
        ))
        
        # Process request
        response = await self.generate_response(request)
        
        # Publish completion event
        await self.event_bus.publish(Event(
            event_type=SystemEvents.AGENT_RESPONSE_SENT,
            payload={
                "agent_id": self.agent_id,
                "request_id": request.request_id,
                "response_length": len(response.content)
            }
        ))
        
        return response

# Other agents subscribe to events
class EmotionAgent:
    async def initialize(self, event_bus: EventBus):
        await event_bus.subscribe(EventSubscription(
            event_type=SystemEvents.AGENT_REQUEST_RECEIVED,
            handler=self.analyze_emotion,
            subscriber_id="emotion_agent"
        ))
    
    async def analyze_emotion(self, event: Event):
        # Analyze emotion and publish results
        emotion_data = self.extract_emotion(event.payload["content"])
        
        await self.event_bus.publish(Event(
            event_type="emotion.analyzed",
            payload={
                "request_id": event.payload["request_id"],
                "emotion": emotion_data
            }
        ))
```

### 5. Error Handling Migration

#### ❌ Old Approach
```python
# Generic exceptions
try:
    response = gemini_client.generate_content(prompt)
except Exception as e:
    logger.error(f"Error: {e}")
    return "Sorry, I encountered an error."
```

#### ✅ New Approach
```python
# Structured exceptions with context
from src.core.exceptions.llm_exceptions import (
    LLMProviderError, LLMRateLimitError, LLMAuthenticationError
)

try:
    response = await self.llm_provider.generate_response(messages)
except LLMRateLimitError as e:
    # Specific handling for rate limits
    retry_after = e.context.get('retry_after_seconds', 60)
    logger.warning(f"Rate limited, retry after {retry_after}s", 
                  extra=e.context)
    return AgentResponse(
        content="I'm temporarily busy, please try again in a moment.",
        success=False,
        error="rate_limited"
    )
except LLMAuthenticationError as e:
    # Specific handling for auth errors
    logger.error("LLM authentication failed", extra=e.context)
    # Don't expose auth details to user
    return AgentResponse(
        content="I'm having technical difficulties.",
        success=False,
        error="service_unavailable"
    )
except LLMProviderError as e:
    # General LLM provider error
    logger.error(f"LLM provider error: {e.message}", extra=e.context)
    return AgentResponse(
        content="I'm having trouble generating a response.",
        success=False,
        error="generation_failed"
    )
```

### 6. Testing Migration

#### ❌ Old Approach
```python
# Hard to test due to tight coupling
def test_chat_agent():
    agent = ChatAgent()  # Creates real dependencies
    response = agent.process("Hello")  # Makes real API calls
    assert "hello" in response.lower()
```

#### ✅ New Approach
```python
# Easy to test with dependency injection
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
async def chat_agent():
    # Mock dependencies
    mock_llm = Mock(spec=LLMInterface)
    mock_llm.generate_response = AsyncMock(return_value=LLMResponse(
        content="Hello! How can I help you?",
        metadata={"provider": "mock"},
        usage={"total_tokens": 10}
    ))
    
    mock_storage = Mock(spec=StorageInterface)
    mock_event_bus = Mock(spec=EventBus)
    
    # Create agent with mocked dependencies
    agent = ChatAgent(
        llm=mock_llm,
        storage=mock_storage,
        event_bus=mock_event_bus
    )
    
    return agent, mock_llm, mock_storage, mock_event_bus

async def test_chat_agent_response(chat_agent):
    agent, mock_llm, mock_storage, mock_event_bus = chat_agent
    
    # Test processing
    request = AgentRequest(content="Hello", context={})
    response = await agent.process_message(request)
    
    # Verify behavior
    assert response.success
    assert response.content == "Hello! How can I help you?"
    
    # Verify interactions
    mock_llm.generate_response.assert_called_once()
    mock_event_bus.publish.assert_called()

async def test_different_providers():
    """Test the same agent with different LLM providers"""
    providers = [
        Mock(spec=LLMInterface, provider_name="mock_gemini"),
        Mock(spec=LLMInterface, provider_name="mock_openai")
    ]
    
    for provider in providers:
        provider.generate_response = AsyncMock(return_value=LLMResponse(
            content=f"Response from {provider.provider_name}",
            metadata={"provider": provider.provider_name}
        ))
        
        agent = ChatAgent(llm=provider, storage=Mock(), event_bus=Mock())
        response = await agent.process_message(
            AgentRequest(content="Test", context={})
        )
        
        assert provider.provider_name in response.content
```

## 🚀 Quick Start with New Architecture

### 1. Application Bootstrap
```python
# main.py
import asyncio
from src.infrastructure.di.container import get_container
from src.infrastructure.config.config_manager import get_config_manager
from src.core.factories.llm_factory import LLMFactory

async def bootstrap_application():
    # 1. Load configuration
    config = get_config_manager()
    config.add_provider(FileConfigProvider("config/app.json"))
    config.add_provider(EnvironmentConfigProvider())
    await config.load_all()
    
    # 2. Setup DI container
    container = get_container()
    
    # 3. Create and register LLM provider
    llm_config = config.get_section("llm")
    llm_provider = await LLMFactory.create_from_config(llm_config)
    container.register_instance(LLMInterface, llm_provider)
    
    # 4. Register other services
    container.register_singleton(StorageInterface, VectorStorageProvider)
    container.register_singleton(EventBus)
    container.register_singleton(ChatAgent)
    
    # 5. Initialize all services
    await container.initialize_all()
    
    return container

async def main():
    container = await bootstrap_application()
    
    # Use the application
    chat_agent = await container.resolve(ChatAgent)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        request = AgentRequest(content=user_input, context={})
        response = await chat_agent.process_message(request)
        print(f"Bot: {response.content}")
    
    # Cleanup
    await container.shutdown_all()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Provider Switching Example
```python
# Switch providers without code changes

# config/development.json
{
    "llm": {
        "provider": "gemini",
        "model_name": "gemini-2.0-flash",
        "api_key": "${GEMINI_API_KEY}"
    }
}

# config/production.json  
{
    "llm": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "api_key": "${OPENAI_API_KEY}"
    }
}

# Load environment-specific config
config_file = f"config/{os.getenv('ENVIRONMENT', 'development')}.json"
config.add_provider(FileConfigProvider(config_file))
```

## ⚠️ Common Migration Issues

### 1. Import Path Changes
```python
# Old imports
from src.models.gemini_llm import GeminiLLM
from src.agents.chat_agent import ChatAgent

# New imports
from src.core.interfaces.llm_interface import LLMInterface
from src.providers.llm.gemini_provider import GeminiProvider
from src.agents.chat_agent import ChatAgent  # Same, but now uses DI
```

### 2. Configuration Format Changes
```python
# Old: Hardcoded in Python
MODEL_NAME = "gemini-2.0-flash"

# New: JSON configuration with validation
{
    "llm": {
        "model_name": "gemini-2.0-flash"
    }
}
```

### 3. Async/Await Usage
```python
# All LLM operations are now async
response = await llm_provider.generate_response(messages)  # await required
```

## 🧪 Testing the Migration

### Unit Tests
```bash
# Test individual components
python -m pytest tests/unit/test_llm_factory.py
python -m pytest tests/unit/test_chat_agent.py

# Test configuration
python -m pytest tests/unit/test_config_manager.py
```

### Integration Tests
```bash
# Test provider switching
python -m pytest tests/integration/test_provider_switching.py

# Test end-to-end flows
python -m pytest tests/integration/test_chat_flow.py
```

### Demo Scripts
```bash
# Run the provider switching demo
python examples/provider_switching_demo.py

# Test configuration management
python examples/config_demo.py
```

## 📈 Performance Impact

The new architecture has minimal performance overhead:

- **DI Container**: Singleton resolution is cached
- **Event System**: Async pub/sub with minimal latency
- **Factory Pattern**: Instance creation overhead only at startup
- **Configuration**: Values are cached after first access

## 🎯 Next Steps

1. **Migrate Core Components**: Start with LLM providers
2. **Update Configuration**: Move to JSON-based config
3. **Add Dependency Injection**: Refactor constructors
4. **Implement Events**: Replace direct method calls
5. **Add Tests**: Leverage dependency injection for mocking
6. **Monitor Performance**: Ensure no regressions

The new architecture provides a **solid foundation** for scaling your mental health chatbot while maintaining high code quality and easy maintenance! 🚀