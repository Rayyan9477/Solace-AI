"""
Logging Module for Contextual Chatbot

This module provides a centralized logging system for the entire application,
with support for structured logging, different output formats, and customizable
log levels per module.
"""

import logging
import os
import sys
import json
import time
import traceback
from typing import Dict, Any, Optional, Union, List
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from src.config.settings import AppConfig

# Constants
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

# Ensure log directory exists
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

# Module level loggers cache
_loggers = {}

class StructuredLogRecord(logging.LogRecord):
    """Extended LogRecord that supports structured data in logs"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structured_data = None


class StructuredLogger(logging.Logger):
    """Logger that supports structured logging with metadata"""
    
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, 
                  func=None, extra=None, sinfo=None):
        """Create a LogRecord with support for structured data"""
        structured_data = None
        
        # Extract structured data if present
        if args and len(args) == 1 and isinstance(args[0], dict):
            structured_data = args[0]
            args = ()
        
        # Create record
        record = super().makeRecord(name, level, fn, lno, msg, args, exc_info, 
                                   func, extra, sinfo)
        
        # Add structured data
        record.structured_data = structured_data
        
        return record


class JSONFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON objects"""
    
    def format(self, record):
        """Format the log record as JSON"""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add structured data if present
        if hasattr(record, "structured_data") and record.structured_data:
            log_data["data"] = record.structured_data
        
        return json.dumps(log_data)


def configure_logger(logger_name: str, log_level: Optional[Union[str, int]] = None,
                    log_file: Optional[str] = None, console: bool = True,
                    json_format: bool = False, rotating: bool = True,
                    max_bytes: int = 10485760, backup_count: int = 5) -> logging.Logger:
    """
    Configure a logger with the specified parameters.
    
    Args:
        logger_name: Name of the logger
        log_level: Logging level (default: from config or INFO)
        log_file: Path to log file (default: logs/{logger_name}.log)
        console: Whether to log to console
        json_format: Whether to use JSON formatting for logs
        rotating: Use rotating file handler instead of standard
        max_bytes: Maximum size for rotating log files
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Get config values
    try:
        config = AppConfig()
        config_log_level = getattr(config, "LOG_LEVEL", None)
    except:
        config_log_level = None
    
    # Determine log level
    if log_level is None:
        log_level = config_log_level or DEFAULT_LOG_LEVEL
    
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), DEFAULT_LOG_LEVEL)
    
    # Create logger instance
    logging.setLoggerClass(StructuredLogger)
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Set formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file is None:
        log_file = os.path.join(DEFAULT_LOG_DIR, f"{logger_name.replace('.', '_')}.log")
    
    if rotating:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
    else:
        file_handler = logging.FileHandler(log_file)
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(module_name: str, config: Dict[str, Any] = None) -> logging.Logger:
    """
    Get a preconfigured logger for a module.
    
    This function caches loggers to ensure only one instance per module exists.
    
    Args:
        module_name: Name of the module requesting the logger
        config: Optional configuration for the logger
        
    Returns:
        Configured logger instance
    """
    global _loggers
    
    # Return cached logger if available
    if module_name in _loggers:
        return _loggers[module_name]
    
    # Default configuration
    logger_config = {
        "log_level": None,  # Use default from config
        "console": True,
        "json_format": False,
        "rotating": True
    }
    
    # Update with provided config
    if config:
        logger_config.update(config)
    
    # Configure logger
    logger = configure_logger(
        module_name,
        log_level=logger_config["log_level"],
        console=logger_config["console"],
        json_format=logger_config["json_format"],
        rotating=logger_config["rotating"]
    )
    
    # Cache the logger
    _loggers[module_name] = logger
    
    return logger


def set_global_log_level(level: Union[str, int]) -> None:
    """
    Set the log level for all configured loggers.
    
    Args:
        level: Log level to set (string or integer)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)
    
    for logger_name, logger in _loggers.items():
        logger.setLevel(level)
        logger.info(f"Log level changed to {logging.getLevelName(level)}")


def get_all_loggers() -> List[str]:
    """Get list of all configured logger names"""
    return list(_loggers.keys())


def configure_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Configure application-wide logging settings
    
    Args:
        log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Determine log level
    level = getattr(logging, log_level.upper()) if log_level else DEFAULT_LOG_LEVEL
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    root_logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_file:
        log_path = os.path.join(DEFAULT_LOG_DIR, log_file)
        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root_logger.addHandler(file_handler)
    
    # Configure default application log file
    app_log_path = os.path.join(DEFAULT_LOG_DIR, "contextual_chatbot.log")
    app_file_handler = RotatingFileHandler(
        app_log_path, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    app_file_handler.setLevel(level)
    app_file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    root_logger.addHandler(app_file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {logging.getLevelName(level)}")


# Create the root application logger
app_logger = get_logger("contextual_chatbot")