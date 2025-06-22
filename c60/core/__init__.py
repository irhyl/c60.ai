"""
Core components of the C60.ai AutoML framework.

This package contains the fundamental building blocks of the AutoML system,
including the main AutoML class, pipeline components, and core utilities.
"""

from .automl import AutoML
from .pipeline import Pipeline
from .evaluator import Evaluator
from .optimizer import Optimizer
from .generator import PipelineGenerator

__all__ = [
    'AutoML',
    'Pipeline',
    'Evaluator',
    'Optimizer',
    'PipelineGenerator'
]
