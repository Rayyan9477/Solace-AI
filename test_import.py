#!/usr/bin/env python3
import sys
import os

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Python path:")
for path in sys.path:
    print(f"  {path}")

try:
    from src.config.settings import AppConfig
    print("SUCCESS: Successfully imported AppConfig")
    print(f"App name: {AppConfig.APP_NAME}")
except Exception as e:
    print(f"ERROR: Failed to import AppConfig: {e}")

try:
    from src.main import Application
    print("SUCCESS: Successfully imported Application")
except Exception as e:
    print(f"ERROR: Failed to import Application: {e}")