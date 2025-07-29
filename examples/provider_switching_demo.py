"""
Demo: Easy LLM Provider Switching

This example demonstrates how the new modular architecture allows
for easy switching between different LLM providers (Gemini, OpenAI, etc.)
through simple configuration changes.
"""

import asyncio
from typing import Dict, Any

# Import the new architecture components
from src.core.factories.llm_factory import LLMFactory, LLMProviderType
from src.core.interfaces.llm_interface import LLMConfig, Message, MessageRole
from src.infrastructure.di.container import DIContainer
from src.infrastructure.config.config_manager import ConfigManager


async def demo_provider_switching():
    """Demonstrate switching between LLM providers."""
    
    print("🤖 LLM Provider Switching Demo")
    print("=" * 50)
    
    # Configuration for different providers
    provider_configs = {
        "gemini": {
            "provider": "gemini",
            "model_name": "gemini-2.0-flash",
            "api_key": "your-gemini-api-key",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "openai": {
            "provider": "openai", 
            "model_name": "gpt-4o",
            "api_key": "your-openai-api-key",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    # Test message
    test_messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful mental health support assistant."
        ),
        Message(
            role=MessageRole.USER,
            content="I'm feeling anxious about an upcoming presentation. Can you help?"
        )
    ]
    
    # Test each provider
    for provider_name, config in provider_configs.items():
        print(f"\n🔄 Testing {provider_name.upper()} Provider")
        print("-" * 30)
        
        try:
            # Create provider using factory
            provider = await LLMFactory.create_from_config(
                config=config,
                instance_id=f"{provider_name}_instance"
            )
            
            print(f"✅ {provider_name} provider initialized successfully")
            
            # Get model info
            model_info = provider.get_model_info()
            print(f"📋 Model: {model_info['model_name']}")
            print(f"🔧 Supports streaming: {model_info['supports_streaming']}")
            
            # Test connection
            if await provider.validate_connection():
                print("🌐 Connection validated")
                
                # Generate response (commented out to avoid API calls in demo)
                # response = await provider.generate_response(test_messages)
                # print(f"💬 Response: {response.content[:100]}...")
                
                print("✨ Provider test completed successfully")
            else:
                print("❌ Connection validation failed")
                
        except Exception as e:
            print(f"❌ Error with {provider_name}: {str(e)}")
    
    # Demonstrate health check
    print(f"\n🏥 Factory Health Check")
    print("-" * 30)
    health_info = await LLMFactory.health_check_all()
    print(f"Total instances: {health_info['total_instances']}")
    print(f"Available providers: {health_info['registered_providers']}")
    
    # Cleanup
    await LLMFactory.shutdown_all_instances()
    print("\n🧹 Cleanup completed")


async def demo_dependency_injection():
    """Demonstrate dependency injection with different providers."""
    
    print("\n🔌 Dependency Injection Demo")
    print("=" * 50)
    
    # Create DI container
    container = DIContainer()
    
    # Register different LLM providers
    gemini_config = LLMConfig(
        model_name="gemini-2.0-flash",
        api_key="your-gemini-api-key",
        temperature=0.7
    )
    
    # Register as singleton (same instance reused)
    container.register_singleton(
        service_type=LLMConfig,
        instance=gemini_config
    )
    
    print("✅ LLM configuration registered in DI container")
    
    # Example service that uses LLM
    class ChatService:
        def __init__(self, llm_config: LLMConfig):
            self.llm_config = llm_config
            self.provider = None
        
        async def initialize(self):
            """Initialize the chat service with LLM provider."""
            try:
                self.provider = await LLMFactory.create_provider(
                    provider_type=LLMProviderType.GEMINI,
                    config=self.llm_config
                )
                return True
            except Exception as e:
                print(f"Failed to initialize chat service: {e}")
                return False
        
        async def chat(self, message: str) -> str:
            """Send a chat message."""
            if not self.provider:
                return "Service not initialized"
            
            messages = [Message(role=MessageRole.USER, content=message)]
            
            try:
                # This would actually call the LLM in production
                return f"Mock response to: {message}"
            except Exception as e:
                return f"Error: {str(e)}"
    
    # Register chat service
    container.register_transient(ChatService)
    
    # Resolve and use the service
    try:
        chat_service = await container.resolve(ChatService)
        
        if await chat_service.initialize():
            print("✅ Chat service initialized with dependency injection")
            
            # Test the service
            response = await chat_service.chat("Hello, how are you?")
            print(f"💬 Chat response: {response}")
        else:
            print("❌ Chat service initialization failed")
            
    except Exception as e:
        print(f"❌ DI Error: {str(e)}")
    
    # Cleanup
    await container.shutdown_all()
    print("🧹 DI container cleanup completed")


async def demo_configuration_management():
    """Demonstrate flexible configuration management."""
    
    print("\n⚙️ Configuration Management Demo")
    print("=" * 50)
    
    config_manager = ConfigManager()
    
    # Example configurations for different environments
    dev_config = {
        "llm": {
            "provider": "gemini",
            "model": "gemini-2.0-flash",
            "temperature": 0.9,  # Higher creativity for development
            "max_tokens": 2000
        },
        "debug": True,
        "logging_level": "DEBUG"
    }
    
    prod_config = {
        "llm": {
            "provider": "openai",  # Different provider in production
            "model": "gpt-4o",
            "temperature": 0.7,    # More conservative in production
            "max_tokens": 1500
        },
        "debug": False,
        "logging_level": "INFO"
    }
    
    # Simulate loading different configs
    current_env = "development"  # Could be from environment variable
    
    if current_env == "development":
        print("📝 Loading development configuration")
        active_config = dev_config
    else:
        print("📝 Loading production configuration")
        active_config = prod_config
    
    # Show how configuration affects provider selection
    llm_config = active_config["llm"]
    print(f"🤖 Selected LLM Provider: {llm_config['provider']}")
    print(f"🧠 Model: {llm_config['model']}")
    print(f"🌡️ Temperature: {llm_config['temperature']}")
    print(f"📊 Debug Mode: {active_config['debug']}")
    
    # This shows how easy it is to switch providers based on config
    provider_type = LLMProviderType(llm_config["provider"])
    print(f"✅ Would initialize {provider_type.value} provider")
    
    print("🎯 Configuration-driven provider selection completed")


async def main():
    """Run all demos."""
    print("🚀 Modular Architecture Demo")
    print("This demo shows the new pluggable architecture")
    print("=" * 60)
    
    # Note: API calls are commented out to avoid requiring actual API keys
    await demo_provider_switching()
    await demo_dependency_injection()
    await demo_configuration_management()
    
    print("\n✨ All demos completed!")
    print("\n🎯 Key Benefits Demonstrated:")
    print("  • Easy provider switching through configuration")
    print("  • Dependency injection for loose coupling")
    print("  • Flexible configuration management")
    print("  • Pluggable architecture with interfaces")
    print("  • Proper error handling and validation")


if __name__ == "__main__":
    asyncio.run(main())