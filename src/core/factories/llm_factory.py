"""
Factory for creating LLM providers.

This factory enables easy switching between different LLM providers
(Gemini, OpenAI, Anthropic, etc.) through configuration.
"""

from typing import Dict, Any, Type, Optional
from enum import Enum
import asyncio

from ..interfaces.llm_interface import LLMInterface, LLMConfig
from ..exceptions.factory_exceptions import ProviderNotFoundError, ProviderInitializationError


class LLMProviderType(Enum):
    """Enum for supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"


class LLMFactory:
    """
    Factory for creating LLM provider instances.
    
    This factory allows for easy registration and creation of different
    LLM providers, enabling pluggable architecture where providers
    can be switched through configuration.
    """
    
    _providers: Dict[LLMProviderType, Type[LLMInterface]] = {}
    _instances: Dict[str, LLMInterface] = {}
    
    @classmethod
    def register_provider(
        self,
        provider_type: LLMProviderType,
        provider_class: Type[LLMInterface]
    ) -> None:
        """
        Register an LLM provider class.
        
        Args:
            provider_type: Type of the provider
            provider_class: Class implementing LLMInterface
        """
        if not issubclass(provider_class, LLMInterface):
            raise TypeError(f"Provider class must implement LLMInterface")
        
        self._providers[provider_type] = provider_class
    
    @classmethod
    def unregister_provider(self, provider_type: LLMProviderType) -> None:
        """
        Unregister an LLM provider.
        
        Args:
            provider_type: Type of the provider to unregister
        """
        if provider_type in self._providers:
            del self._providers[provider_type]
    
    @classmethod
    def get_available_providers(self) -> list[LLMProviderType]:
        """
        Get list of available provider types.
        
        Returns:
            List of available provider types
        """
        return list(self._providers.keys())
    
    @classmethod
    async def create_provider(
        self,
        provider_type: LLMProviderType,
        config: LLMConfig,
        instance_id: Optional[str] = None
    ) -> LLMInterface:
        """
        Create an LLM provider instance.
        
        Args:
            provider_type: Type of provider to create
            config: Configuration for the provider
            instance_id: Optional unique ID for the instance
            
        Returns:
            Initialized LLM provider instance
            
        Raises:
            ProviderNotFoundError: If provider type is not registered
            ProviderInitializationError: If provider initialization fails
        """
        if provider_type not in self._providers:
            raise ProviderNotFoundError(
                f"LLM provider '{provider_type.value}' is not registered. "
                f"Available providers: {[p.value for p in self._providers.keys()]}"
            )
        
        provider_class = self._providers[provider_type]
        
        try:
            # Create instance
            provider = provider_class(config)
            
            # Initialize the provider
            if not await provider.initialize():
                raise ProviderInitializationError(
                    f"Failed to initialize {provider_type.value} provider"
                )
            
            # Store instance if ID provided
            if instance_id:
                self._instances[instance_id] = provider
            
            return provider
            
        except Exception as e:
            if isinstance(e, (ProviderNotFoundError, ProviderInitializationError)):
                raise
            raise ProviderInitializationError(
                f"Error creating {provider_type.value} provider: {str(e)}"
            ) from e
    
    @classmethod
    async def create_from_config(
        self,
        config: Dict[str, Any],
        instance_id: Optional[str] = None
    ) -> LLMInterface:
        """
        Create an LLM provider from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing provider details
            instance_id: Optional unique ID for the instance
            
        Returns:
            Initialized LLM provider instance
            
        Example config:
        {
            "provider": "gemini",
            "model_name": "<from env>",
            "api_key": "your-api-key",
            "temperature": 0.7,
            "max_tokens": 2000
        }
        """
        provider_type_str = config.get("provider")
        if not provider_type_str:
            raise ProviderInitializationError("Provider type not specified in config")
        
        try:
            provider_type = LLMProviderType(provider_type_str.lower())
        except ValueError:
            raise ProviderNotFoundError(
                f"Unknown provider type: {provider_type_str}. "
                f"Available: {[p.value for p in LLMProviderType]}"
            )
        
        # Create LLMConfig from dictionary
        llm_config = LLMConfig(
            model_name=config.get("model_name", ""),
            api_key=config.get("api_key", ""),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            top_p=config.get("top_p", 0.9),
            top_k=config.get("top_k"),
            additional_params=config.get("additional_params", {})
        )
        
        return await self.create_provider(provider_type, llm_config, instance_id)
    
    @classmethod
    def get_instance(self, instance_id: str) -> Optional[LLMInterface]:
        """
        Get a stored provider instance by ID.
        
        Args:
            instance_id: ID of the instance to retrieve
            
        Returns:
            Provider instance if found, None otherwise
        """
        return self._instances.get(instance_id)
    
    @classmethod
    def remove_instance(self, instance_id: str) -> bool:
        """
        Remove a stored provider instance.
        
        Args:
            instance_id: ID of the instance to remove
            
        Returns:
            True if instance was removed, False if not found
        """
        if instance_id in self._instances:
            del self._instances[instance_id]
            return True
        return False
    
    @classmethod
    async def shutdown_all_instances(self) -> None:
        """Shutdown all stored provider instances."""
        shutdown_tasks = []
        for instance in self._instances.values():
            if hasattr(instance, 'shutdown'):
                shutdown_tasks.append(instance.shutdown())
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self._instances.clear()
    
    @classmethod
    def get_provider_info(self, provider_type: LLMProviderType) -> Dict[str, Any]:
        """
        Get information about a registered provider.
        
        Args:
            provider_type: Type of provider to get info for
            
        Returns:
            Dictionary containing provider information
        """
        if provider_type not in self._providers:
            return {"error": "Provider not registered"}
        
        provider_class = self._providers[provider_type]
        return {
            "provider_type": provider_type.value,
            "class_name": provider_class.__name__,
            "module": provider_class.__module__,
            "doc": provider_class.__doc__
        }
    
    @classmethod
    async def health_check_all(self) -> Dict[str, Any]:
        """
        Perform health check on all provider instances.
        
        Returns:
            Dictionary containing health status of all instances
        """
        health_results = {}
        
        for instance_id, provider in self._instances.items():
            try:
                health_info = await provider.health_check()
                health_results[instance_id] = health_info
            except Exception as e:
                health_results[instance_id] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "total_instances": len(self._instances),
            "registered_providers": [p.value for p in self._providers.keys()],
            "instances": health_results
        }


# Auto-register built-in providers when module is imported
def _register_builtin_providers():
    """Register built-in LLM providers."""
    try:
        # Try to register Gemini provider
        from ...providers.llm.gemini_provider import GeminiProvider
        LLMFactory.register_provider(LLMProviderType.GEMINI, GeminiProvider)
    except ImportError:
        pass
    
    try:
        # Try to register OpenAI provider
        from ...providers.llm.openai_provider import OpenAIProvider
        LLMFactory.register_provider(LLMProviderType.OPENAI, OpenAIProvider)
    except ImportError:
        pass
    
    try:
        # Try to register Anthropic provider
        from ...providers.llm.anthropic_provider import AnthropicProvider
        LLMFactory.register_provider(LLMProviderType.ANTHROPIC, AnthropicProvider)
    except ImportError:
        pass
    
    try:
        # Try to register HuggingFace provider
        from ...providers.llm.huggingface_provider import HuggingFaceProvider
        LLMFactory.register_provider(LLMProviderType.HUGGINGFACE, HuggingFaceProvider)
    except ImportError:
        pass
    
    try:
        # Try to register Ollama provider
        from ...providers.llm.ollama_provider import OllamaProvider
        LLMFactory.register_provider(LLMProviderType.OLLAMA, OllamaProvider)
    except ImportError:
        pass


# Register providers when module is loaded
_register_builtin_providers()