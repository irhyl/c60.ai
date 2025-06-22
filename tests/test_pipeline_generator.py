"""
Tests for the pipeline_generator module.

This module contains unit tests for the heuristic-based pipeline generation.
"""

import pytest
import pandas as pd
import numpy as np
from engine.pipeline_generator import PipelineGenerator
from engine.graph_schema import PipelineGraph


def test_generate_baseline_pipeline():
    """Test generating a baseline pipeline."""
    # Create a sample dataset
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    # Initialize generator
    generator = PipelineGenerator()
    
    # Generate pipeline
    pipeline = generator.generate_baseline_pipeline(X, 'target')
    
    # Verify pipeline structure
    assert isinstance(pipeline, PipelineGraph)
    assert len(pipeline.graph.nodes) > 0
    assert len(pipeline.graph.edges) > 0


def test_feature_selection_heuristic():
    """Test the feature selection heuristic."""
    # Create a sample dataset with informative and noisy features
    np.random.seed(42)
    X = pd.DataFrame({
        'informative1': np.random.rand(100),
        'informative2': np.random.rand(100),
        'noise1': np.random.rand(100),
        'noise2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    generator = PipelineGenerator()
    selected = generator._select_features_heuristic(X, 'target', n_features=2)
    
    # Should select 2 features (excluding target)
    assert len(selected) == 2
    assert 'target' not in selected


def test_model_selection_heuristic_classification():
    """Test model selection for classification tasks."""
    generator = PipelineGenerator()
    model_type = generator._select_model_heuristic('classification', 1000, 10)
    assert model_type in ['random_forest', 'xgboost', 'logistic_regression']


def test_model_selection_heuristic_regression():
    """Test model selection for regression tasks."""
    generator = PipelineGenerator()
    model_type = generator._select_model_heuristic('regression', 1000, 10)
    assert model_type in ['random_forest', 'xgboost', 'linear_regression']


def test_feature_engineering_heuristic():
    """Test feature engineering heuristic."""
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    generator = PipelineGenerator()
    steps = generator._get_feature_engineering_steps(X, 'target')
    
    # Should return a list of transformation steps
    assert isinstance(steps, list)
    assert all(isinstance(step, dict) for step in steps)


def test_handle_missing_values():
    """Test missing value handling."""
    X = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [np.nan, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })
    
    generator = PipelineGenerator()
    result = generator._handle_missing_values(X, 'target')
    
    # Should not contain any NaN values
    assert not result.isna().any().any()
    assert 'target' in result.columns
