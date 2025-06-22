"""
Tests for the scorer module.

This module contains unit tests for the pipeline scoring functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from c60.engine.scorer import PipelineScorer
from c60.engine.graph_schema import PipelineGraph


class TestPipelineScorer:
    """Test suite for the PipelineScorer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y = np.random.randint(0, 2, 100)
        return X, y

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        pipeline = MagicMock()
        pipeline.fit.return_value = pipeline
        pipeline.predict.return_value = np.random.rand(100)
        pipeline.predict_proba.return_value = np.column_stack([
            np.random.rand(100), np.random.rand(100)
        ])
        return pipeline

    def test_score_classification(self, sample_data, mock_pipeline):
        """Test scoring for classification tasks."""
        X, y = sample_data
        scorer = PipelineScorer(task_type='classification')
        
        scores = scorer.score(mock_pipeline, X, y)
        
        assert 'accuracy' in scores
        assert 'f1' in scores
        assert 'precision' in scores
        assert 'recall' in scores
        assert 'roc_auc' in scores
        assert all(0 <= v <= 1 for v in scores.values())

    def test_score_regression(self, sample_data, mock_pipeline):
        """Test scoring for regression tasks."""
        X, y = sample_data
        y = y.astype(float)  # Convert to float for regression
        scorer = PipelineScorer(task_type='regression')
        
        scores = scorer.score(mock_pipeline, X, y)
        
        assert 'r2' in scores
        assert 'mse' in scores
        assert 'mae' in scores
        assert 'explained_variance' in scores

    def test_cross_validation(self, sample_data, mock_pipeline):
        """Test cross-validation scoring."""
        X, y = sample_data
        scorer = PipelineScorer(task_type='classification', cv=3)
        
        scores = scorer.score(mock_pipeline, X, y)
        
        assert 'cv_accuracy' in scores
        assert 'cv_f1' in scores
        assert 'cv_precision' in scores
        assert 'cv_recall' in scores

    def test_custom_metrics(self, sample_data, mock_pipeline):
        """Test scoring with custom metrics."""
        X, y = sample_data
        
        def custom_metric(y_true, y_pred):
            return 0.5  # Dummy metric
            
        scorer = PipelineScorer(
            task_type='classification',
            metrics={'custom': custom_metric}
        )
        
        scores = scorer.score(mock_pipeline, X, y)
        assert 'custom' in scores
        assert scores['custom'] == 0.5

    def test_feature_importance(self, sample_data, mock_pipeline):
        """Test feature importance calculation."""
        X, y = sample_data
        mock_pipeline.feature_importances_ = np.array([0.7, 0.3])
        
        scorer = PipelineScorer(task_type='classification', calculate_feature_importance=True)
        scores = scorer.score(mock_pipeline, X, y)
        
        assert 'feature_importances' in scores
        assert len(scores['feature_importances']) == 2
        assert all(isinstance(x, float) for x in scores['feature_importances'])

    def test_invalid_task_type(self):
        """Test initialization with invalid task type."""
        with pytest.raises(ValueError, match="Unsupported task type"):
            PipelineScorer(task_type='invalid_task')
