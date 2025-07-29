# 🏗️ Contextual-Chatbot Architecture

## Overview

The Contextual-Chatbot has been redesigned with a **modular, scalable, and maintainable architecture** that follows modern software engineering principles. The new architecture enables easy extensibility, provider switching, and testing.

## 🎯 Key Principles

### 1. **Separation of Concerns**
- **Interfaces** define contracts
- **Implementations** provide concrete functionality  
- **Factories** handle object creation
- **DI Container** manages dependencies

### 2. **Dependency Inversion** 
- High-level modules don't depend on low-level modules
- Both depend on abstractions (interfaces)
- Easy to swap implementations

### 3. **Open/Closed Principle**
- Open for extension (new providers, agents)
- Closed for modification (core interfaces stable)

### 4. **Single Responsibility**
- Each class has one reason to change
- Clear boundaries between components

## 📁 Architecture Structure

```
src/
├── core/                           # Core abstractions and contracts
│   ├── interfaces/                 # Abstract interfaces
│   │   ├── llm_interface.py       # LLM provider contract
│   │   ├── agent_interface.py     # Agent contract
│   │   ├── storage_interface.py   # Storage contract
│   │   └── config_interface.py    # Configuration contract
│   ├── factories/                  # Factory patterns
│   │   ├── llm_factory.py         # LLM provider factory
│   │   └── agent_factory.py       # Agent factory
│   └── exceptions/                 # Structured exceptions
├── providers/                      # Concrete implementations
│   ├── llm/                       # LLM providers
│   │   ├── gemini_provider.py     # Google Gemini
│   │   ├── openai_provider.py     # OpenAI GPT
│   │   └── anthropic_provider.py  # Anthropic Claude
│   ├── storage/                   # Storage providers
│   └── voice/                     # Voice providers  
├── infrastructure/                 # Cross-cutting concerns
│   ├── di/                        # Dependency injection
│   ├── config/                    # Configuration management
│   └── logging/                   # Structured logging
└── agents/                        # Business logic agents
    ├── chat_agent.py              # Refactored with DI
    ├── therapy_agent.py           # Composition-based
    └── diagnosis_agent.py         # Event-driven
```

## 🔌 Provider System

### Easy LLM Switching

Switch between LLM providers with **zero code changes**:

```python
# Configuration-driven provider selection
config = {
    "provider": "gemini",           # Change to "openai", "anthropic"
    "model_name": "gemini-2.0-flash",
    "api_key": "your-api-key",
    "temperature": 0.7
}

# Factory creates the right provider
provider = await LLMFactory.create_from_config(config)
```

### Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| **Google Gemini** | 2.0 Flash, Pro | ✅ Implemented |
| **OpenAI** | GPT-4o, GPT-3.5 | ✅ Implemented |
| **Anthropic** | Claude 3.5 Sonnet | 🚧 Planned |
| **HuggingFace** | Local models | 🚧 Planned |
| **Ollama** | Local hosting | 🚧 Planned |

## 🏭 Factory Pattern

### LLM Factory

```python
from src.core.factories.llm_factory import LLMFactory, LLMProviderType

# Create provider by type
provider = await LLMFactory.create_provider(
    provider_type=LLMProviderType.GEMINI,
    config=llm_config,
    instance_id="main_llm"
)

# Create from configuration
provider = await LLMFactory.create_from_config({
    "provider": "openai",
    "model_name": "gpt-4o", 
    "api_key": "sk-..."
})

# Health check all instances
health = await LLMFactory.health_check_all()
```

## 💉 Dependency Injection

### Container Usage

```python
from src.infrastructure.di import DIContainer

container = DIContainer()

# Register services
container.register_singleton(ILLMProvider, GeminiProvider)
container.register_transient(IChatAgent, ChatAgent) 

# Auto-resolve dependencies
chat_agent = await container.resolve(IChatAgent)
```

### Service Decorators

```python
from src.infrastructure.di.decorators import service, inject

@service(IChatService, lifecycle="singleton")
class ChatService:
    def __init__(self, llm: ILLMProvider, storage: IStorage):
        self.llm = llm
        self.storage = storage

@inject(IChatService)
async def handle_chat(service: IChatService, message: str):
    return await service.process_message(message)
```

## ⚙️ Configuration System

### Multi-Source Configuration

```python
from src.infrastructure.config import ConfigManager

config = ConfigManager()

# Add providers in priority order
config.add_provider(EnvironmentConfigProvider(), priority=100)
config.add_provider(FileConfigProvider("config.json"), priority=50)
config.add_provider(DatabaseConfigProvider(), priority=10)

# Get values (checks providers in priority order)
llm_provider = config.get_value("llm.provider", default="gemini")
debug_mode = config.get_value("debug", default=False, value_type=bool)
```

### Schema Validation

```python
from src.infrastructure.config.schema import ConfigSchema, ConfigField

schema = ConfigSchema("llm_config", fields=[
    ConfigField("provider", str, required=True),
    ConfigField("model_name", str, required=True),
    ConfigField("temperature", float, default=0.7, min_value=0.0, max_value=2.0),
    ConfigField("max_tokens", int, default=2000, min_value=1, max_value=8192)
])

config.register_schema("llm", schema)
validation = config.validate_schema("llm")
```

## 🔄 Event-Driven Architecture

### Event System

```python
from src.core.interfaces.event_interface import EventBus, Event, SystemEvents

event_bus = EventBus()
await event_bus.start()

# Publish events
await event_bus.publish(Event(
    event_type=SystemEvents.LLM_REQUEST_STARTED,
    payload={"provider": "gemini", "model": "2.0-flash"}
))

# Subscribe to events
async def handle_llm_events(event: Event):
    print(f"LLM Event: {event.event_type}")

await event_bus.subscribe(EventSubscription(
    event_type=SystemEvents.LLM_REQUEST_STARTED,
    handler=handle_llm_events,
    subscriber_id="llm_monitor"
))
```

## 🧩 Agent Composition

### Old vs New Approach

#### ❌ Old (Inheritance-Heavy)
```python
class ChatAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.llm = GeminiLLM()  # Tight coupling
        self.storage = VectorStore()  # Hard to test
        
    def process(self, message):
        # 500+ lines of mixed concerns
        pass
```

#### ✅ New (Composition-Based)
```python
@service(IChatAgent, lifecycle="singleton")
class ChatAgent:
    def __init__(self, 
                 llm: ILLMProvider,
                 storage: IStorage,
                 events: IEventBus):
        self.llm = llm           # Injected dependency
        self.storage = storage   # Easy to mock for testing
        self.events = events     # Event-driven communication
    
    async def process_message(self, request: AgentRequest) -> AgentResponse:
        # Focused, single responsibility
        await self.events.publish(create_agent_event(
            SystemEvents.AGENT_REQUEST_RECEIVED,
            self.agent_id,
            request_id=request.request_id
        ))
        
        # Process with injected dependencies
        response = await self.llm.generate_response(messages)
        await self.storage.store_conversation(request, response)
        
        return AgentResponse(
            content=response.content,
            confidence=0.9,
            metadata={"provider": self.llm.provider_name}
        )
```

## 🧪 Testing Benefits

### Easy Mocking

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
async def chat_agent():
    # Mock dependencies
    mock_llm = Mock(spec=ILLMProvider)
    mock_storage = Mock(spec=IStorage)
    mock_events = Mock(spec=IEventBus)
    
    # Inject mocks
    agent = ChatAgent(mock_llm, mock_storage, mock_events)
    await agent.initialize()
    return agent

async def test_chat_agent_processing(chat_agent):
    # Test with mocked dependencies
    request = AgentRequest(content="Hello", context={})
    response = await chat_agent.process_message(request)
    
    assert response.success
    assert response.content is not None
```

## 🚀 Getting Started

### 1. Basic Setup

```python
import asyncio
from src.core.factories.llm_factory import LLMFactory, LLMProviderType
from src.infrastructure.di import DIContainer, get_container
from src.infrastructure.config import ConfigManager

async def setup_application():
    # 1. Setup configuration
    config = ConfigManager()
    await config.load_all()
    
    # 2. Setup DI container
    container = get_container()
    
    # 3. Register LLM provider
    llm_config = config.get_section("llm")
    llm_provider = await LLMFactory.create_from_config(llm_config)
    container.register_instance(ILLMProvider, llm_provider)
    
    # 4. Register agents
    container.register_singleton(IChatAgent, ChatAgent)
    
    # 5. Initialize all services
    await container.initialize_all()
    
    return container

# Usage
container = await setup_application()
chat_agent = await container.resolve(IChatAgent)
response = await chat_agent.process_message("Hello!")
```

### 2. Adding New Provider

```python
# 1. Implement the interface
class CustomLLMProvider(LLMInterface):
    async def generate_response(self, messages):
        # Your implementation
        pass

# 2. Register with factory  
LLMFactory.register_provider(
    LLMProviderType.CUSTOM,
    CustomLLMProvider
)

# 3. Use through configuration
config = {
    "provider": "custom",
    "model_name": "custom-model",
    # ... other settings
}
provider = await LLMFactory.create_from_config(config)
```

## 📊 Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Provider Switching** | Code changes required | Configuration change only |
| **Testing** | Difficult to mock | Easy dependency injection |
| **Coupling** | Tight coupling | Loose coupling via interfaces |
| **Extensibility** | Modify existing code | Add new implementations |
| **Configuration** | Hardcoded values | Flexible, validated config |
| **Error Handling** | Generic exceptions | Structured, typed errors |
| **Monitoring** | Basic logging | Event-driven telemetry |

## 🎯 Real-World Usage

### Switching Providers by Environment

```python
# Development: Use local/free provider
DEV_CONFIG = {
    "llm": {"provider": "ollama", "model": "llama2"}
}

# Production: Use enterprise provider  
PROD_CONFIG = {
    "llm": {"provider": "openai", "model": "gpt-4o"}
}

# Code stays the same, behavior changes via config
provider = await LLMFactory.create_from_config(
    config=get_config_for_environment()
)
```

### A/B Testing Different Models

```python
# Test different providers for the same user query
providers = ["gemini", "openai", "anthropic"]

responses = []
for provider_name in providers:
    config = get_provider_config(provider_name)  
    provider = await LLMFactory.create_from_config(config)
    
    response = await provider.generate_response(messages)
    responses.append({
        "provider": provider_name,
        "response": response.content,
        "confidence": calculate_confidence(response)
    })

# Choose best response
best_response = max(responses, key=lambda r: r["confidence"])
```

## 🔮 Future Enhancements

1. **Plugin System**: Hot-swappable functionality
2. **Circuit Breakers**: Automatic failover between providers
3. **Caching Layer**: Intelligent response caching
4. **Metrics Collection**: Comprehensive observability
5. **Auto-scaling**: Dynamic provider selection based on load

---

This architecture provides a **solid foundation** for a production-ready mental health chatbot that can adapt to changing requirements, scale efficiently, and maintain high code quality. 🚀