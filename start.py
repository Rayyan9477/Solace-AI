#!/usr/bin/env python3
"""
Startup script for Contextual-Chatbot with enhanced error handling.

This script initializes the application with proper error handling,
encoding configuration, and environment validation.
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

try:
    # Configure environment and Unicode handling first
    from src.utils.console_utils import setup_console, safe_print

    # Set up console for Unicode support
    setup_console()
    
    # Now import other dependencies
    import asyncio
    from src.config.settings import AppConfig
    from src.utils.logger import configure_logging, get_logger
    from app import ChatbotCLI
    from src.main import Application, initialize_components
    
    # Configure logging
    configure_logging(AppConfig.LOG_LEVEL)
    logger = get_logger("startup")
    
    def check_environment():
        """Verify that the environment is properly configured"""
        # Check Python version
        py_version = sys.version_info
        if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
            safe_print("Warning: This application requires Python 3.9 or higher.")
            safe_print(f"Current Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
        
        # Check for required API keys
        if not AppConfig.GEMINI_API_KEY:
            safe_print("Error: GEMINI_API_KEY not found in environment variables or .env file.")
            safe_print("Please set this API key and try again.")
            return False
        
        # Check for required directories
        data_dir = script_dir / "data"
        if not data_dir.exists():
            safe_print(f"Creating data directory at {data_dir}")
            os.makedirs(data_dir, exist_ok=True)
        
        # Verify paths in configuration
        for path_name in ["VECTOR_DB_CONFIG", "STORAGE_PATH", "DATA_DIR"]:
            if hasattr(AppConfig, path_name):
                path_value = getattr(AppConfig, path_name)
                if isinstance(path_value, dict) and "storage_path" in path_value:
                    path_str = path_value["storage_path"]
                    if not os.path.exists(path_str):
                        safe_print(f"Creating directory for {path_name}: {path_str}")
                        os.makedirs(path_str, exist_ok=True)
        
        return True
    
    async def start_application():
        """Initialize and start the application"""
        try:
            # Initialize application components
            app = initialize_components()
            
            if not app or not app.initialized:
                safe_print("Error: Application initialization failed.")
                return False
            
            # Create CLI interface
            cli = ChatbotCLI(app)
            
            # Start interactive session
            await cli.start_interactive_session()
            
            # Graceful shutdown
            await app.shutdown()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting application: {str(e)}")
            traceback.print_exc()
            return False
    
    def main():
        """Main entry point"""
        safe_print("Starting Contextual-Chatbot...")
        
        # Check environment
        if not check_environment():
            safe_print("Environment check failed. Exiting.")
            sys.exit(1)
        
        # Run the application
        success = asyncio.run(start_application())
        
        if not success:
            safe_print("Application did not exit successfully.")
            sys.exit(1)
        
        safe_print("Application exited successfully.")
    
    if __name__ == "__main__":
        main()
        
except Exception as e:
    # Handle startup errors without logger
    print(f"Critical startup error: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
