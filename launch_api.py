"""
API Server Launcher

This script launches the FastAPI server for the Contextual-Chatbot application.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

def launch_api_server(port=8000, reload=True):
    """
    Launch the FastAPI server
    
    Args:
        port: Port to run the server on
        reload: Whether to enable auto-reload
    """
    print("Starting API server...")
    cmd = [
        "uvicorn", 
        "api_server:app", 
        "--host", "0.0.0.0", 
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    launch_api_server(port=args.port, reload=not args.no_reload)
