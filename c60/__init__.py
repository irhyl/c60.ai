"""
C60.ai - Generative Evolutionary AutoML Framework

C60.ai is an advanced AutoML framework that leverages generative and evolutionary
algorithms to automate the machine learning pipeline development process.
"""

__version__ = "0.1.0"

# Core imports
from .core.automl import AutoML
from .core.pipeline import Pipeline

# Utility imports
from .utils.config import Config
from .utils.logging import get_logger

# Export main classes
__all__ = [
    'AutoML',
    'Pipeline',
    'Config',
    'get_logger'
]
