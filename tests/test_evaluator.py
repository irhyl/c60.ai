"""
Tests for the Evaluator class.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from c60.core.evaluator import Evaluator


class TestEvaluator:
    """Test cases for the Evaluator class."""

    @pytest.fixture
    def classification_data(self):
        """Generate synthetic classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Generate synthetic regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=5,
            noise=0.1,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def classification_model(self):
        """Create a simple classification model."""
        return RandomForestClassifier(n_estimators=10, random_state=42)

    @pytest.fixture
    def regression_model(self):
        """Create a simple regression model."""
        return RandomForestRegressor(n_estimators=10, random_state=42)

    def test_evaluate_classification(self, classification_data, classification_model):
        """Test evaluation of a classification model."""
        X, y = classification_data
        evaluator = Evaluator(task='classification')
        
        # Test with default metrics
        scores = evaluator.evaluate(classification_model, X, y)
        
        # Check that scores are returned for all default metrics
        assert 'accuracy' in scores
        assert 'f1_weighted' in scores
        assert 'roc_auc_ovr' in scores
        
        # Check that scores are within expected ranges
        assert 0 <= scores['accuracy'] <= 1
        assert 0 <= scores['f1_weighted'] <= 1
        assert 0 <= scores['roc_auc_ovr'] <= 1

    def test_evaluate_regression(self, regression_data, regression_model):
        """Test evaluation of a regression model."""
        X, y = regression_data
        evaluator = Evaluator(task='regression')
        
        # Test with default metrics
        scores = evaluator.evaluate(regression_model, X, y)
        
        # Check that scores are returned for all default metrics
        assert 'neg_mean_squared_error' in scores
        assert 'r2' in scores
        assert 'neg_mean_absolute_error' in scores
        
        # Check that MSE is negative (as expected for neg_mean_squared_error)
        assert scores['neg_mean_squared_error'] <= 0
        
        # R2 should be between -inf and 1
        assert scores['r2'] <= 1

    def test_custom_metrics(self, classification_data, classification_model):
        """Test evaluation with custom metrics."""
        X, y = classification_data
        custom_metrics = ['precision', 'recall', 'f1']
        evaluator = Evaluator(task='classification', metrics=custom_metrics)
        
        scores = evaluator.evaluate(classification_model, X, y)
        
        # Check that only the custom metrics are returned
        assert set(scores.keys()) == set(custom_metrics)
        
        # Check that all scores are within expected ranges
        for metric in custom_metrics:
            assert 0 <= scores[metric] <= 1

    def test_holdout_evaluation(self, classification_data, classification_model):
        """Test holdout evaluation with validation set."""
        from sklearn.model_selection import train_test_split
        
        X, y = classification_data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        evaluator = Evaluator(task='classification')
        scores = evaluator.evaluate(
            classification_model, 
            X_train, y_train, 
            X_val, y_val
        )
        
        # Check that scores are returned
        assert 'accuracy' in scores
        assert 0 <= scores['accuracy'] <= 1

    def test_invalid_metric(self, classification_data, classification_model):
        """Test evaluation with an invalid metric."""
        X, y = classification_data
        evaluator = Evaluator(task='classification', metrics=['invalid_metric'])
        
        with pytest.raises(ValueError):
            evaluator.evaluate(classification_model, X, y)

    def test_cross_validation(self, classification_data, classification_model):
        """Test cross-validation evaluation."""
        X, y = classification_data
        evaluator = Evaluator(task='classification', cv=5)
        
        scores = evaluator.evaluate(classification_model, X, y)
        
        # Check that scores are returned with CV suffix
        assert 'accuracy' in scores
        assert 'accuracy_std' in scores
        assert 0 <= scores['accuracy'] <= 1
        assert scores['accuracy_std'] >= 0
