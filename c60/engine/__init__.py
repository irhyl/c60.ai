"""
C60 Engine Module

This module contains the core logic for the C60 AutoML pipeline system.
It includes components for data loading, pipeline generation, optimization,
and execution.

Modules:
    data_loader: Handles dataset loading and validation
    graph_schema: Defines the pipeline graph structure
    graph_validator: Validates pipeline graphs
    pipeline_generator: Generates ML pipelines
    scorer: Scores pipeline performance
    optimizer: Optimizes pipeline parameters
    cache_manager: Manages caching for efficiency
    feedback_loop: Handles user feedback integration
"""

__version__ = "0.1.0"

# Import core components
from . import data_loader
from . import graph_schema
from . import graph_validator

__all__ = [
    'data_loader',
    'graph_schema',
    'graph_validator',
                    ]
