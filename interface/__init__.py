"""
C60 Interface Module

This module provides user interfaces for interacting with the C60 AutoML system,
including command-line and API interfaces.

Modules:
    cli: Command-line interface for user interaction
    api: REST API for cloud integration
    cloud_runner: Cloud execution logic
"""

__version__ = "0.1.0"

# Import interface components
from . import cli
from . import api
from . import cloud_runner

__all__ = [
    'cli',
    'api',
    'cloud_runner',
]
