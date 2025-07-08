#!/usr/bin/env python3
import os
from pathlib import Path

# Check for .env file
env_file = Path('.env')
print(f".env file exists: {env_file.exists()}")

if env_file.exists():
    print(f".env file size: {env_file.stat().st_size} bytes")
    with open(env_file, 'r') as f:
        content = f.read()
        print(f"Content preview: {content[:100]}...")

# Check environment variable
api_key = os.environ.get('GEMINI_API_KEY')
if api_key:
    print(f"GEMINI_API_KEY is set (length: {len(api_key)})")
else:
    print("GEMINI_API_KEY is NOT set")

# Try loading with dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
    api_key_after_load = os.environ.get('GEMINI_API_KEY')
    if api_key_after_load:
        print(f"After dotenv load - GEMINI_API_KEY is set (length: {len(api_key_after_load)})")
    else:
        print("After dotenv load - GEMINI_API_KEY is still NOT set")
except ImportError:
    print("python-dotenv not installed")