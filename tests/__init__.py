"""
Test suite for C60 AutoML system.

This package contains all the tests for the C60 AutoML system,
including unit tests, integration tests, and system tests.
"""
__version__ = "0.1.0"

# Import test modules to make them discoverable by pytest
from . import test_data_loader
from . import test_graph_schema
from . import test_graph_validator
from . import test_pipeline_generator
from . import test_scorer
from . import test_optimizer
from . import test_cache_manager
from . import test_feedback_loop
from . import test_api
from . import test_cli
from . import test_cloud_runner

__all__ = [
    'test_data_loader',
    'test_graph_schema',
    'test_graph_validator',
    'test_pipeline_generator',
    'test_scorer',
    'test_optimizer',
    'test_cache_manager',
    'test_feedback_loop',
    'test_api',
    'test_cli',
    'test_cloud_runner',
]
