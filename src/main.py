"""
Main application entry point for the Contextual-Chatbot.

This module initializes the application, sets up the module system,
configures logging, and starts the user interface.
"""

import streamlit as st
import asyncio
import time
from datetime import datetime
import os
import sys
from pathlib import Path
import torch

# Configure import paths
sys.path.append(str(Path(__file__).resolve().parent))

# Import settings first to ensure configuration is loaded
from src.config.settings import AppConfig

# Import core modules
from src.components.base_module import Module, ModuleManager, get_module_manager
from src.utils.logger import get_logger, configure_logging
from src.utils.metrics import track_metric

# Initialize logger
logger = get_logger(__name__)

# Configure CUDA/CPU device
def setup_device():
    """Configure and return the appropriate device (CUDA or CPU)"""
    if torch.cuda.is_available():
        try:
            # Test CUDA initialization
            torch.cuda.init()
            device = "cuda"
            logger.info("CUDA is available and initialized successfully")
        except Exception as e:
            device = "cpu"
            logger.warning(f"CUDA initialization failed, falling back to CPU: {str(e)}")
    else:
        device = "cpu"
        logger.info("CUDA is not available, using CPU")
    
    return device

class Application:
    """
    Main application class that handles initialization, module management,
    and application lifecycle.
    """
    
    def __init__(self):
        """Initialize the application"""
        self.module_manager = get_module_manager()
        self.initialized = False
        self.device = setup_device()
        self.logger = get_logger(__name__)
        
        # Environment validation
        self._validate_environment()
        
        # Configure logging based on app settings
        self._configure_logging()
        
        self.logger.info(f"Application initialized with device: {self.device}")
    
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
        module_count = self.module_manager.discover_module_types("src.agents")
        self.logger.info(f"Discovered {module_count} agent modules")
        
        module_count += self.module_manager.discover_module_types("src.components")
        self.logger.info(f"Discovered {module_count} total modules")
        
        # Create core modules with configurations from AppConfig
        self._create_core_modules()
        
        # Initialize all modules
        success = await self.module_manager.initialize_all()
        self.initialized = success
        
        if success:
            self.logger.info("Application initialization complete")
        else:
            self.logger.error("Application initialization failed")
        
        return success
    
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
        
        # Create agent modules with dependencies
        agent_configs = {
            "safety": {},
            "emotion": {},
            "chat": {},
            "diagnosis": {},
            "personality": {},
            "crawler": AppConfig.get_crawler_config(),
            "search": {},
            "integrated_diagnosis": {}
        }
        
        # Create agent modules
        for agent_name, agent_config in agent_configs.items():
            self.module_manager.create_module(
                type_id=f"{agent_name.capitalize()}Agent",
                module_id=f"{agent_name}_agent",
                config=agent_config
            )
        
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
        
        # Log health status
        if health_info["overall_status"] == "operational":
            self.logger.info("Health check: All systems operational", 
                         {"modules": len(health_info["modules"]), "initialized": health_info["initialized_modules"]})
        else:
            self.logger.warning("Health check: System degraded", 
                            {"modules": len(health_info["modules"]), "initialized": health_info["initialized_modules"]})
        
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

            # Start the UI
            self.start_ui()
            
        except Exception as e:
            self.logger.critical(f"Fatal error in application: {str(e)}")
            raise
    
    def migrate_data(self):
        """Migrate data to the central vector database"""
        from utils.migration_utils import migrate_all_user_data
        
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

    def start_ui(self):
        """Start the user interface"""
        main()

def main():
    """Main entry point for the application"""
    # Run through Streamlit
    if "app" not in st.session_state:
        # Initialize application
        app = Application()
        st.session_state["app"] = app
        
        # Initialize session variables
        if "step" not in st.session_state:
            reset_session()
        
        # Initialize modules asynchronously
        with st.spinner("Initializing application modules..."):
            success = asyncio.run(app.initialize())
            
            if not success:
                st.error("Failed to initialize all application modules. Some features may be unavailable.")
                logger.error("Application initialization failed in main()")
    
    # Get application instance
    app = st.session_state["app"]
    
    # Set up Streamlit UI
    st.title(AppConfig.APP_NAME)
    st.markdown("### A Safe Space for Mental Health Support")
    
    # Show system status
    with st.expander("System Status", expanded=False):
        # Get latest health check
        health_info = asyncio.run(app.health_check())
        
        # Display status
        st.markdown("### Component Status")
        
        for module_id, module_info in health_info["modules"].items():
            status = module_info["status"]
            if status == "operational":
                st.success(f"✅ {module_id}: Available")
            else:
                st.error(f"❌ {module_id}: {status}")
        
        # Show environment info
        st.markdown("### Environment")
        st.write(f"App Version: {AppConfig.APP_VERSION}")
        st.write(f"Debug Mode: {'Enabled' if AppConfig.DEBUG else 'Disabled'}")
        st.write(f"Model: {AppConfig.MODEL_NAME}")
        st.write(f"Device: {app.device}")
    
    # Get UI manager
    ui_manager = app.module_manager.get_module("ui_manager")
    
    if ui_manager and ui_manager.initialized:
        # Render the appropriate UI based on the session state
        asyncio.run(ui_manager.render_ui(st.session_state["step"]))
    else:
        st.error("UI Manager not available. Application cannot continue.")
        logger.error("UI Manager not initialized or not found")

def reset_session():
    """Reset the application session state"""
    st.session_state.clear()
    st.session_state.update({
        "step": 1,
        "symptoms": [],
        "diagnosis": "",
        "personality": {},
        "history": [],
        "start_time": time.time(),
        "assessment_component": None,
        "assessment_complete": False,
        "assessment_data": {},
        "integrated_assessment_results": {},
        "empathy_response": "",
        "immediate_actions": [],
        "metrics": {
            "interactions": 0,
            "response_times": [],
            "safety_flags": 0
        }
    })
    logger.info("Session reset")

def initialize_components(config=None):
    """
    Initialize application components

    Args:
        config: Optional configuration override

    Returns:
        Application instance
    """
    try:
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
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please check the logs or contact support.")