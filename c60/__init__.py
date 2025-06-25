"""
C60.ai - Generative Evolutionary AutoML Framework

Main package entrypoint for the C60.ai AutoML framework.
Exposes core AutoML and pipeline classes, configuration, and logging utilities.
"""

__version__ = "0.1.0"

from .core.automl import AutoML
from .core.pipeline import Pipeline
from .utils.config import Config
from .utils.logging import get_logger

__all__ = [
    'AutoML',
    'Pipeline',
    'Config',
    'get_logger',
]
