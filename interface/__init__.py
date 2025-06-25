"""
C60 Interface Module

Provides user-facing interfaces for the C60 AutoML system, including:
- Command-line interface (CLI)
- REST API for cloud integration
- Cloud execution logic
"""

__version__ = "0.1.0"

from . import cli
from . import api
from . import cloud_runner

__all__ = [
    'cli',
    'api',
    'cloud_runner',
]
