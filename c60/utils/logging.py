"""
Logging utilities for the C60 AutoML framework.

This module provides a centralized logging system for the C60 AutoML framework.
It configures logging with sensible defaults and provides utility functions
for getting loggers with consistent formatting.
"""

import logging
import os
import sys
from typing import Optional

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Configure root logger
def configure_logging(
    level: int = logging.INFO,
    format: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    filename: Optional[str] = None,
    filemode: str = 'a',
) -> None:
    """
    Configure the root logger with the specified settings.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format: Log message format
        datefmt: Date format string
        filename: Optional file to log to. If None, logs to stderr.
        filemode: File mode if filename is specified ('a' for append, 'w' for write)
    """
    handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    handlers.append(console_handler)
    
    # Add file handler if filename is provided
    if filename:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        file_handler = logging.FileHandler(filename, mode=filemode)
        file_handler.setLevel(level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format,
        datefmt=datefmt,
        handlers=handlers
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    If no name is provided, returns the root logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: int) -> None:
    """
    Set the log level for all handlers of the root logger.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


# Initialize with default configuration when module is imported
configure_logging()

# Export public API
__all__ = ['get_logger', 'configure_logging', 'set_log_level']
