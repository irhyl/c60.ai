"""
Tests for the Pipeline class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from c60.core.pipeline import Pipeline, PipelineStep


class TestPipeline:
    """Test cases for the Pipeline class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_classes=2,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def sample_pipeline(self):
        """Create a sample pipeline for testing."""
        pipeline = Pipeline()
        pipeline.add_step(
            name="imputer",
            estimator=SimpleImputer(strategy="mean")
        )
        pipeline.add_step(
            name="scaler",
            estimator=StandardScaler()
        )
        pipeline.add_step(
            name="classifier",
            estimator=RandomForestClassifier(n_estimators=10, random_state=42)
        )
        return pipeline

    def test_add_step(self):
        """Test adding steps to the pipeline."""
        pipeline = Pipeline()
        assert len(pipeline.steps) == 0
        
        # Add first step
        pipeline.add_step("step1", SimpleImputer())
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "step1"
        
        # Add second step
        pipeline.add_step("step2", StandardScaler())
        assert len(pipeline.steps) == 2
        assert pipeline.steps[1].name == "step2"

    def test_remove_step(self, sample_pipeline):
        """Test removing steps from the pipeline."""
        assert len(sample_pipeline.steps) == 3
        
        # Remove middle step
        sample_pipeline.remove_step("scaler")
        assert len(sample_pipeline.steps) == 2
        assert sample_pipeline.steps[0].name == "imputer"
        assert sample_pipeline.steps[1].name == "classifier"
        
        # Try to remove non-existent step (should not raise error)
        sample_pipeline.remove_step("nonexistent")
        assert len(sample_pipeline.steps) == 2

    def test_get_step(self, sample_pipeline):
        """Test getting a step by name."""
        step = sample_pipeline.get_step("scaler")
        assert step is not None
        assert step.name == "scaler"
        assert isinstance(step.estimator, StandardScaler)
        
        # Test getting non-existent step
        assert sample_pipeline.get_step("nonexistent") is None

    def test_set_step_params(self, sample_pipeline):
        """Test setting parameters for a step."""
        # Set parameters for the classifier step
        sample_pipeline.set_step_params(
            "classifier",
            n_estimators=50,
            max_depth=5
        )
        
        # Verify parameters were set
        classifier = sample_pipeline.get_step("classifier").estimator
        assert classifier.n_estimators == 50
        assert classifier.max_depth == 5

    def test_fit_predict(self, sample_pipeline, sample_data):
        """Test fitting the pipeline and making predictions."""
        X, y = sample_data
        
        # Split data
        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]
        
        # Fit and predict
        sample_pipeline.fit(X_train, y_train)
        y_pred = sample_pipeline.predict(X_test)
        
        # Verify predictions
        assert len(y_pred) == len(X_test)
        assert set(y_pred).issubset({0, 1})  # Binary classification

    @patch('sklearn.pipeline.Pipeline.fit')
    def test_fit_calls_underlying_pipeline(self, mock_fit, sample_pipeline, sample_data):
        """Test that fit calls the underlying scikit-learn pipeline."""
        X, y = sample_data
        sample_pipeline.fit(X, y)
        mock_fit.assert_called_once()

    @patch('sklearn.pipeline.Pipeline.predict')
    def test_predict_calls_underlying_pipeline(self, mock_predict, sample_pipeline, sample_data):
        """Test that predict calls the underlying scikit-learn pipeline."""
        X, _ = sample_data
        mock_predict.return_value = np.zeros(len(X))
        
        sample_pipeline.predict(X)
        mock_predict.assert_called_once()

    def test_pipeline_repr(self, sample_pipeline):
        """Test the string representation of the pipeline."""
        repr_str = repr(sample_pipeline)
        assert "Pipeline(" in repr_str
        assert "steps=" in repr_str
        assert "imputer" in repr_str
        assert "scaler" in repr_str
        assert "classifier" in repr_str

    def test_get_params(self, sample_pipeline):
        """Test getting parameters from the pipeline."""
        params = sample_pipeline.get_params()
        
        # Check that parameters from all steps are included
        assert 'imputer__strategy' in params
        assert 'classifier__n_estimators' in params
        
        # Check parameter values
        assert params['classifier__n_estimators'] == 10

    def test_set_params(self, sample_pipeline):
        """Test setting parameters on the pipeline."""
        # Set parameters
        sample_pipeline.set_params(
            classifier__n_estimators=100,
            classifier__max_depth=10
        )
        
        # Verify parameters were set
        classifier = sample_pipeline.get_step("classifier").estimator
        assert classifier.n_estimators == 100
        assert classifier.max_depth == 10
