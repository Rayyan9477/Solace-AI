"""
Mental Health Support Chatbot Application
Main entry point for the command-line and API application
"""

import asyncio
import logging
import argparse
import sys
from typing import Dict, Any, Optional
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import application components
from src.main import Application, initialize_components
from src.config.settings import AppConfig
from src.utils.logger import get_logger, configure_logging
from src.utils.device_utils import get_device_info, is_cuda_available
from src.utils.console_utils import setup_console, safe_print, emoji_to_ascii

# Set up console for Unicode
setup_console()

# Configure logging
configure_logging(AppConfig.LOG_LEVEL)
logger = get_logger(__name__)

class ChatbotCLI:
    """Command-line interface for the chatbot"""
    
    def __init__(self, app: Application):
        self.app = app
        self.logger = get_logger(__name__)
        
    async def start_interactive_session(self):
        """Start an interactive chat session"""
        welcome_msg = emoji_to_ascii(f"\n** Welcome to {AppConfig.APP_NAME}")
        safe_print(welcome_msg)
        safe_print("Your Empathetic Digital Confidant")
        safe_print("-" * 50)
        
        # Display system information
        device_info = get_device_info()
        safe_print(f"Device: {device_info['device']}")
        safe_print(f"CUDA Available: {device_info['cuda_available']}")
        if device_info['using_cuda']:
            safe_print(f"GPU: {device_info.get('gpu_name', 'Unknown')}")
            safe_print(f"GPU Memory: {device_info.get('gpu_memory_total', 0):.2f} GB")
        safe_print("-" * 50)
        
        safe_print("\nType 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for available commands.\n")
        
        # Get chat agent from module manager
        chat_agent = self.app.module_manager.get_module("chat_agent")
        if not chat_agent:
            print("❌ Chat agent not available. Please check your configuration.")
            return
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thank you for chatting. Take care!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'status':
                    await self._show_status()
                    continue
                elif user_input.lower() == 'clear':
                    conversation_history.clear()
                    print("🔄 Conversation history cleared.")
                    continue
                
                # Process the message
                print("🤔 Thinking...")
                
                response = await chat_agent.process_message(
                    user_input, 
                    conversation_history=conversation_history
                )
                
                if response.get('success', False):
                    assistant_message = response.get('response', 'I apologize, but I had trouble processing your message.')
                    print(f"\nAssistant: {assistant_message}\n")
                    
                    # Update conversation history
                    conversation_history.append({
                        'user': user_input,
                        'assistant': assistant_message,
                        'timestamp': time.time()
                    })
                    
                    # Keep only last 10 exchanges
                    if len(conversation_history) > 10:
                        conversation_history = conversation_history[-10:]
                        
                else:
                    error_msg = response.get('error', 'Unknown error occurred')
                    print(f"\n❌ Error: {error_msg}\n")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error in chat session: {str(e)}")
                print(f"\n❌ An error occurred: {str(e)}\n")
    
    def _show_help(self):
        """Show available commands"""
        print("\n📋 Available Commands:")
        print("  help     - Show this help message")
        print("  status   - Show system status")
        print("  clear    - Clear conversation history")
        print("  quit     - Exit the chatbot")
        print()
    
    async def _show_status(self):
        """Show system status"""
        health_info = await self.app.health_check()
        
        print("\n📊 System Status:")
        print(f"Overall Status: {'✅ Operational' if health_info['overall_status'] == 'operational' else '⚠️ Degraded'}")
        print(f"Initialized Modules: {health_info['initialized_modules']}/{len(health_info['modules'])}")
        
        print("\nModule Status:")
        for module_id, module_info in health_info["modules"].items():
            status = module_info["status"]
            if status == "operational":
                print(f"  ✅ {module_id}")
            else:
                print(f"  ❌ {module_id}: {status}")
        
        device_info = get_device_info()
        print(f"\nDevice: {device_info['device']}")
        if device_info['using_cuda']:
            print(f"GPU Memory Used: {device_info.get('gpu_memory_allocated', 0):.2f} GB")
        print()

async def run_migration(user_id: str = "default_user"):
    """Run data migration"""
    print("🔄 Starting data migration...")
    app = Application()
    
    try:
        from src.utils.migration_utils import migrate_all_user_data
        results = migrate_all_user_data(user_id=user_id)
        
        total = sum(results.values())
        print(f"✅ Migration complete: {total} total items migrated.")
        for data_type, count in results.items():
            if count > 0:
                print(f"  - {data_type}: {count} items")
                
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        print(f"❌ Migration failed: {str(e)}")

async def run_health_check():
    """Run system health check"""
    print("🔍 Running system health check...")
    
    try:
        app = await initialize_components()
        health_info = await app.health_check()
        
        print(f"\n📊 Health Check Results:")
        print(f"Overall Status: {'✅ Operational' if health_info['overall_status'] == 'operational' else '⚠️ Degraded'}")
        print(f"Modules: {health_info['initialized_modules']}/{len(health_info['modules'])} operational")
        
        for module_id, module_info in health_info["modules"].items():
            status = module_info["status"]
            if status == "operational":
                print(f"  ✅ {module_id}")
            else:
                print(f"  ❌ {module_id}: {status}")
        
        device_info = get_device_info()
        print(f"\nDevice Info:")
        print(f"  Device: {device_info['device']}")
        print(f"  CUDA Available: {device_info['cuda_available']}")
        if device_info['using_cuda']:
            print(f"  GPU: {device_info.get('gpu_name', 'Unknown')}")
            print(f"  GPU Memory: {device_info.get('gpu_memory_total', 0):.2f} GB")
        
        await app.shutdown()
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        print(f"❌ Health check failed: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Mental Health Support Chatbot")
    
    # Add command options
    parser.add_argument("--migrate-data", action="store_true", 
                       help="Migrate existing data to central vector database")
    parser.add_argument("--user-id", default="default_user",
                       help="User ID for data migration")
    parser.add_argument("--health-check", action="store_true",
                       help="Run system health check")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug mode if requested
    if args.debug:
        configure_logging("DEBUG")
        logger.info("Debug mode enabled")
    
    # Handle different commands
    if args.migrate_data:
        asyncio.run(run_migration(args.user_id))
        return
        
    if args.health_check:
        asyncio.run(run_health_check())
        return
    
    # Default: start interactive chat
    try:
        app = Application()
        
        # Initialize the application
        async def run_app():
            if not await app.initialize():
                logger.critical("Failed to initialize application")
                print("❌ Failed to initialize application. Check logs for details.")
                return
            
            # Start CLI interface
            cli = ChatbotCLI(app)
            await cli.start_interactive_session()
            
            # Shutdown
            await app.shutdown()
        
        asyncio.run(run_app())
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        print(f"❌ A critical error occurred: {str(e)}")

if __name__ == "__main__":
    main()
