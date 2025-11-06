"""
Main application entry point for the Contextual-Chatbot.

This module initializes the application, sets up the module system,
configures logging, and provides a command-line interface.
"""

import asyncio
import time
from datetime import datetime
import os
import sys
from pathlib import Path
import torch
import logging

# Import paths configured through package structure

# Import settings first to ensure configuration is loaded
from .config.settings import AppConfig

# Import core modules
from .components.base_module import Module, ModuleManager, get_module_manager
from .utils.logger import get_logger, configure_logging
from .utils.metrics import track_metric
from .utils.device_utils import get_device, get_device_info, is_cuda_available
from .utils.console_utils import setup_console, safe_print, emoji_to_ascii

# Set up console for Unicode
setup_console()

# Initialize logger
logger = get_logger(__name__)

class Application:
    """
    Main application class that handles initialization, module management,
    and application lifecycle.
    """
    
    def __init__(self):
        """Initialize the application"""
        self.module_manager = get_module_manager()
        self.initialized = False
        self.device = get_device()
        self.device_info = get_device_info()
        self.logger = get_logger(__name__)
        self.performance_profiler = None
        self.optimization_enabled = False

        # Environment validation
        self._validate_environment()

        # Configure logging based on app settings
        self._configure_logging()

        # Initialize performance profiling if optimization is available
        self._initialize_optimization_components()

        self.logger.info(f"Application initialized with device: {self.device}")
        self.logger.info(f"Device info: {self.device_info}")
        if self.optimization_enabled:
            self.logger.info("Optimization components enabled")
    
    def _configure_logging(self):
        """Configure logging based on application settings"""
        log_config = {
            "log_level": AppConfig.LOG_LEVEL,
            "log_to_file": True,
            "log_to_console": True,
            "json_logging": AppConfig.DEBUG  # Use structured JSON logging in debug mode
        }
        configure_logging(log_config)
        self.logger.info("Logging configured", {"config": log_config})

    def _initialize_optimization_components(self):
        """Initialize optimization and performance profiling components"""
        try:
            # Check if optimization module is available
            from src.agents.orchestration import OPTIMIZATION_ENABLED
            self.optimization_enabled = OPTIMIZATION_ENABLED

            if self.optimization_enabled:
                # Initialize performance profiler
                from src.optimization.performance_profiler import AgentPerformanceProfiler
                self.performance_profiler = AgentPerformanceProfiler()
                self.logger.info("Performance profiler initialized")

                # Log optimization details
                from src.optimization import __version__ as opt_version
                self.logger.info(f"Optimization module version: {opt_version}")

        except ImportError as e:
            self.logger.warning(f"Optimization components not available: {e}")
            self.optimization_enabled = False

    def _validate_environment(self):
        """Validate that required environment variables and configuration are set"""
        # Ensure required configuration is present
        if not hasattr(AppConfig, 'MAX_RESPONSE_TOKENS'):
            setattr(AppConfig, 'MAX_RESPONSE_TOKENS', 2000)
            self.logger.warning("MAX_RESPONSE_TOKENS not found in config, defaulting to 2000")

        if not AppConfig.GEMINI_API_KEY:
            error_msg = "GEMINI_API_KEY is missing from your .env file. Please set it and restart the app."
            self.logger.critical(error_msg)
            raise EnvironmentError(error_msg)
        
        # Validate the configuration
        if not AppConfig.validate_config():
            error_msg = "Invalid configuration detected"
            self.logger.critical(error_msg)
            raise ValueError(error_msg)
    
    async def initialize(self):
        """Initialize all application modules"""
        self.logger.info("Starting application initialization")
        
        # Discover module types in packages
        try:
            module_count = self.module_manager.discover_module_types("src.agents")
            self.logger.info(f"Discovered {module_count} agent modules")
            
            module_count += self.module_manager.discover_module_types("src.components")
            self.logger.info(f"Discovered {module_count} total modules")
            
            # Manual registration of core modules as a fallback
            self._register_core_module_types()
            
            # Create core modules with configurations from AppConfig
            self._create_core_modules()
            
            # Initialize all modules, with graceful degradation
            success = await self._initialize_modules_with_fallback()
            self.initialized = success
            
            if success:
                self.logger.info("Application initialization complete")
            else:
                self.logger.error("Application initialization failed")
            
            return success
        except Exception as e:
            self.logger.error(f"Error during application initialization: {str(e)}")
            self.initialized = False
            return False
    
    def _register_core_module_types(self):
        """Manually register core module types as a fallback"""
        try:
            # Try to import the modules
            try:
                from src.components.llm_module import LLMModule
                self.module_manager.register_module_type(LLMModule)
                self.logger.debug("Registered LLMModule type")
            except ImportError:
                self.logger.warning("Could not import LLMModule")
                
            try:
                from src.components.voice_module import VoiceModule
                self.module_manager.register_module_type(VoiceModule)
                self.logger.debug("Registered VoiceModule type")
            except ImportError:
                self.logger.warning("Could not import VoiceModule")
                
            try:
                from src.components.vector_store_module import VectorStoreModule
                self.module_manager.register_module_type(VectorStoreModule)
                self.logger.debug("Registered VectorStoreModule type")
            except ImportError:
                self.logger.warning("Could not import VectorStoreModule")
                
            try:
                from src.components.central_vector_db_module import CentralVectorDBModule
                self.module_manager.register_module_type(CentralVectorDBModule)
                self.logger.debug("Registered CentralVectorDBModule type")
            except ImportError:
                self.logger.warning("Could not import CentralVectorDBModule")
                
            try:
                from src.components.ui_manager import UIManager
                self.module_manager.register_module_type(UIManager)
                self.logger.debug("Registered UIManager type")
            except ImportError:
                self.logger.warning("Could not import UIManager")
                
        except Exception as e:
            self.logger.error(f"Error registering core module types: {str(e)}")
    
    async def _initialize_modules_with_fallback(self):
        """Initialize modules with fallback for non-critical modules"""
        # First, initialize essential modules
        essential_modules = ["llm", "central_vector_db", "vector_store"]
        optional_modules = ["voice", "ui_manager"]
        
        # Try to initialize essential modules
        all_success = True
        for module_id in essential_modules:
            module = self.module_manager.get_module(module_id)
            if module:
                try:
                    success = await module.initialize()
                    if not success:
                        self.logger.error(f"Failed to initialize essential module: {module_id}")
                        all_success = False
                except Exception as e:
                    self.logger.error(f"Error initializing module {module_id}: {str(e)}")
                    all_success = False
            else:
                self.logger.error(f"Essential module not found: {module_id}")
                all_success = False
        
        # Try to initialize optional modules
        for module_id in optional_modules:
            module = self.module_manager.get_module(module_id)
            if module:
                try:
                    success = await module.initialize()
                    if not success:
                        self.logger.warning(f"Failed to initialize optional module: {module_id}")
                except Exception as e:
                    self.logger.warning(f"Error initializing optional module {module_id}: {str(e)}")
        
        # Initialize remaining modules
        for module_id, module in self.module_manager.modules.items():
            if module_id not in essential_modules and module_id not in optional_modules:
                try:
                    success = await module.initialize()
                    if not success:
                        self.logger.warning(f"Failed to initialize module: {module_id}")
                except Exception as e:
                    self.logger.warning(f"Error initializing module {module_id}: {str(e)}")
        
        return all_success
    
    def _create_core_modules(self):
        """Create core system modules"""
        # Create LLM module
        self.module_manager.create_module(
            type_id="LLMModule",
            module_id="llm",
            config=AppConfig.get_model_config()
        )
        
        # Create Voice AI module if voice settings are present
        if hasattr(AppConfig, 'VOICE_CONFIG'):
            self.module_manager.create_module(
                type_id="VoiceModule",
                module_id="voice",
                config=AppConfig.VOICE_CONFIG
            )
        
        # Create vector store module
        self.module_manager.create_module(
            type_id="VectorStoreModule",
            module_id="vector_store",
            config=AppConfig.VECTOR_DB_CONFIG
        )
        
        # Create central vector database module
        self.module_manager.create_module(
            type_id="CentralVectorDBModule",
            module_id="central_vector_db",
            config={
                "user_id": AppConfig.USER_ID if hasattr(AppConfig, 'USER_ID') else "default_user",
                **AppConfig.VECTOR_DB_CONFIG
            }
        )
        
        # Create orchestrator FIRST (needed for agent coordination)
        # Configure with optimization settings
        orchestrator_config = {
            "enhanced_features_enabled": True,
            "event_driven_enabled": True,
            "supervision_mesh_enabled": True,
            "circuit_breaker_enabled": True,
            "performance_monitoring_enabled": True,
            "retry_config": {
                "max_retries": 3,
                "base_delay": 1.0,
                "max_delay": 60.0
            },
            "circuit_breaker_config": {
                "failure_threshold": 5,
                "reset_timeout": 60,
                "success_threshold": 2,
                "timeout": 30
            }
        }

        orchestrator = self.module_manager.create_module(
            type_id="AgentOrchestrator",
            module_id="orchestrator",
            config=orchestrator_config
        )

        # Create agent modules with dependencies
        # Use optimized model configurations if available
        agent_configs = {}

        # Configure each agent with optimized model settings
        for agent_name in ["safety", "emotion", "chat", "diagnosis", "personality",
                         "therapy", "search", "integrated_diagnosis"]:
            # Get optimized model config for this agent
            if hasattr(AppConfig, 'get_optimized_model_config'):
                model_config = AppConfig.get_optimized_model_config(f"{agent_name}_agent")
            else:
                model_config = AppConfig.get_model_config()

            agent_configs[agent_name] = {"model_config": model_config}

        # Special configuration for crawler agent
        agent_configs["crawler"] = {
            **AppConfig.get_crawler_config(),
            "model_config": AppConfig.get_optimized_model_config("crawler_agent")
            if hasattr(AppConfig, 'get_optimized_model_config') else AppConfig.get_model_config()
        }

        # Create agent modules
        for agent_name, agent_config in agent_configs.items():
            agent = self.module_manager.create_module(
                type_id=f"{agent_name.capitalize()}Agent",
                module_id=f"{agent_name}_agent",
                config=agent_config
            )
            # Register each agent with orchestrator for workflow coordination
            if orchestrator and agent:
                try:
                    orchestrator.register_agent(f"{agent_name}_agent", agent)
                    self.logger.info(f"Registered {agent_name}_agent with orchestrator")
                    if self.optimization_enabled:
                        self.logger.debug(f"Using optimized model for {agent_name}_agent: "
                                       f"{agent_config.get('model_config', {}).get('model', 'default')}")
                except Exception as e:
                    self.logger.warning(f"Could not register {agent_name}_agent: {e}")

        # Create UI module
        self.module_manager.create_module(
            type_id="UIManager",
            module_id="ui_manager",
            config={}
        )
    
    async def shutdown(self):
        """Shutdown all application modules and perform cleanup"""
        self.logger.info("Application shutdown initiated")
        
        if not self.initialized:
            self.logger.warning("Application shutdown called but application was not fully initialized")
        
        # Shutdown all modules in reverse dependency order
        success = await self.module_manager.shutdown_all(reverse_order=True)
        
        if success:
            self.logger.info("Application shutdown complete")
        else:
            self.logger.error("Application shutdown encountered errors")
        
        return success
    
    async def health_check(self):
        """Perform health check on all application modules"""
        health_info = await self.module_manager.health_check_all()

        # Add optimization metrics if available
        if self.optimization_enabled and self.performance_profiler:
            try:
                # Get performance recommendations
                recommendations = self.performance_profiler.get_optimization_recommendations()
                health_info["optimization"] = {
                    "enabled": True,
                    "recommendations": recommendations,
                    "bottlenecks_detected": len(recommendations.get("bottlenecks", [])),
                    "cache_opportunities": len(recommendations.get("cache_opportunities", []))
                }
            except Exception as e:
                self.logger.warning(f"Error getting optimization metrics: {e}")
                health_info["optimization"] = {"enabled": True, "error": str(e)}
        else:
            health_info["optimization"] = {"enabled": False}

        # Log health status
        if health_info["overall_status"] == "operational":
            self.logger.info("Health check: All systems operational",
                         {"modules": len(health_info["modules"]),
                          "initialized": health_info["initialized_modules"],
                          "optimization": health_info["optimization"]["enabled"]})
        else:
            self.logger.warning("Health check: System degraded",
                            {"modules": len(health_info["modules"]),
                             "initialized": health_info["initialized_modules"],
                             "optimization": health_info["optimization"]["enabled"]})

        return health_info

    def run(self):
        """Run the application"""
        try:
            # Check if we need to migrate data
            if "--migrate-data" in sys.argv:
                self.migrate_data()
                return
            
            # Initialize asyncio loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Initialize application
            if not loop.run_until_complete(self.initialize()):
                self.logger.critical("Failed to initialize application")
                return

            # Start the application
            self.start_application()
            
        except Exception as e:
            self.logger.critical(f"Fatal error in application: {str(e)}")
            raise
    
    def migrate_data(self):
        """Migrate data to the central vector database"""
        from .utils.migration_utils import migrate_all_user_data
        
        self.logger.info("Starting data migration to central vector database...")
        
        # Get user ID from command line or use default
        user_id = next((arg.split('=')[1] for arg in sys.argv if arg.startswith("--user-id=")), "default_user")
        
        try:
            # Run migration
            results = migrate_all_user_data(user_id=user_id)
            
            # Log results
            total = sum(results.values())
            self.logger.info(f"Migration complete: {total} total items migrated.")
            for data_type, count in results.items():
                self.logger.info(f"  - {data_type}: {count} items")
                
        except Exception as e:
            self.logger.error(f"Error during migration: {str(e)}")
            
        self.logger.info("Migration process finished.")

    def start_application(self):
        """Start the application without UI dependencies"""
        print(f"Starting {AppConfig.APP_NAME}")
        print("A Safe Space for Mental Health Support")
        print("-" * 50)
        
        # Run health check
        health_info = asyncio.run(self.health_check())
        
        # Display status
        print("System Status:")
        for module_id, module_info in health_info["modules"].items():
            status = module_info["status"]
            if status == "operational":
                print(f"✅ {module_id}: Available")
            else:
                print(f"❌ {module_id}: {status}")
        
        print("\nEnvironment Info:")
        print(f"App Version: {AppConfig.APP_VERSION}")
        print(f"Debug Mode: {'Enabled' if AppConfig.DEBUG else 'Disabled'}")
        print(f"Model: {AppConfig.MODEL_NAME}")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        # Keep application running
        print("Application is running. Press Ctrl+C to exit.")
        try:
            # Main application loop - can be expanded for specific functionality
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down application...")
            asyncio.run(self.shutdown())
            print("Application shutdown complete.")

def main():
    """Main entry point for the application"""
    try:
        # Create and run application
        app = Application()
        app.run()
    except Exception as e:
        logger.critical(f"Unhandled exception in main application: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred: {str(e)}")
        print("Please check the logs or contact support.")

def initialize_components(config=None):
    """
    Initialize application components

    Args:
        config: Optional configuration override

    Returns:
        Application instance
    """
    try:
        # Get module manager
        module_manager = get_module_manager()
        
        # Discover module types in components and agents packages
        comp_count = module_manager.discover_module_types("src.components")
        agent_count = module_manager.discover_module_types("src.agents")
        
        logger.info(f"Discovered {comp_count} component modules and {agent_count} agent modules")
        
        # Ensure we're discovering modules correctly - manually register core modules
        try:
            from .components.llm_module import LLMModule
            from .components.voice_module import VoiceModule
            from .components.vector_store_module import VectorStoreModule
            from .components.central_vector_db_module import CentralVectorDBModule
            from .components.ui_manager import UIManager
            
            module_manager.register_module_type(LLMModule)
            module_manager.register_module_type(VoiceModule)
            module_manager.register_module_type(VectorStoreModule)
            module_manager.register_module_type(CentralVectorDBModule)
            module_manager.register_module_type(UIManager)
            
            # Register any agent types that may have been missed
            try:
                from .agents.chat_agent import ChatAgent
                from .agents.emotion_agent import EmotionAgent
                from .agents.diagnosis_agent import DiagnosisAgent
                from .agents.therapy_agent import TherapyAgent
                from .agents.safety_agent import SafetyAgent
                
                module_manager.register_module_type(ChatAgent)
                module_manager.register_module_type(EmotionAgent)
                module_manager.register_module_type(DiagnosisAgent)
                module_manager.register_module_type(TherapyAgent)
                module_manager.register_module_type(SafetyAgent)
            except ImportError as e:
                logger.warning(f"Some agent modules could not be imported: {e}")
        except ImportError as e:
            logger.warning(f"Some core modules could not be imported: {e}")
        
        # Create and initialize the application
        app = Application()
        asyncio.run(app.initialize())
        return app
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception in main application: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred: {str(e)}")
        print("Please check the logs or contact support.")