#!/usr/bin/env python3
"""
Contextual-Chatbot (Solace AI) - Main Entry Point

A unified entry point for the mental health support chatbot that supports:
- CLI chat interface
- API server mode  
- Environment checking
- Data migration
- Health checks

Usage:
    python main.py                    # Start CLI chat interface
    python main.py --mode api         # Start API server
    python main.py --mode check       # Check environment
    python main.py --migrate-data     # Migrate data
    python main.py --health-check     # Run health check
"""

import argparse
import asyncio
import sys
from pathlib import Path

def main():
    """Main entry point with unified command-line interface"""
    parser = argparse.ArgumentParser(
        description="Contextual-Chatbot (Solace AI) - Mental Health Support Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Start interactive CLI chat
  %(prog)s --mode api           Start API server on port 8000
  %(prog)s --mode check         Check environment and dependencies
  %(prog)s --migrate-data       Migrate existing data to central vector DB
  %(prog)s --health-check       Run comprehensive health check
  %(prog)s --debug              Enable debug logging
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["cli", "api", "check"], 
        default="cli",
        help="Application mode (default: cli)"
    )
    
    # Special operations
    parser.add_argument(
        "--migrate-data", 
        action="store_true",
        help="Migrate existing data to central vector database"
    )
    parser.add_argument(
        "--health-check", 
        action="store_true",
        help="Run system health check"
    )
    
    # Configuration options
    parser.add_argument(
        "--user-id", 
        default="default_user",
        help="User ID for data migration (default: default_user)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port for API server mode (default: 8000)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host for API server mode (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Handle special operations first
    if args.migrate_data:
        from app import run_migration
        asyncio.run(run_migration(args.user_id))
        return
        
    if args.health_check:
        from app import run_health_check
        asyncio.run(run_health_check())
        return
    
    # Handle different modes
    if args.mode == "cli":
        # Start CLI chat interface
        from app import main as run_cli
        run_cli()
        
    elif args.mode == "api":
        # Start API server
        import uvicorn
        uvicorn.run(
            "api_server:app", 
            host=args.host, 
            port=args.port, 
            reload=args.debug,
            log_level="debug" if args.debug else "info"
        )
        
    elif args.mode == "check":
        # Run environment check
        print("Environment check functionality - checking basic imports...")
        try:
            from src.config.settings import AppConfig
            print("Config: OK")
            from src.components.base_module import ModuleManager
            print("Module system: OK") 
            from src.models.llm import GeminiLLM
            print("LLM interface: OK")
            print("Basic environment check completed successfully")
        except Exception as e:
            print(f"Environment check failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)