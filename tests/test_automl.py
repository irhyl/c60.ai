"""
Tests for the AutoML class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, call
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from c60.core.automl import AutoML
from c60.core.pipeline import Pipeline


class TestAutoML:
    """Test cases for the AutoML class."""

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
    def mock_pipeline_generator(self):
        """Create a mock pipeline generator."""
        with patch('c60.core.automl.PipelineGenerator') as mock_class:
            mock_generator = mock_class.return_value
            
            # Create a simple mock pipeline
            mock_pipeline = MagicMock(spec=Pipeline)
            mock_pipeline.fit.return_value = mock_pipeline
            mock_pipeline.predict.return_value = np.array([0, 1, 0, 1])
            
            # Set up the generator to return a list of mock pipelines
            mock_generator.generate_initial_population.return_value = [mock_pipeline] * 3
            
            yield mock_generator

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        with patch('c60.core.automl.Evaluator') as mock_class:
            mock_evaluator = mock_class.return_value
            mock_evaluator.evaluate.return_value = {'score': 0.9}
            yield mock_evaluator

    def test_init_defaults(self):
        """Test AutoML initialization with default parameters."""
        automl = AutoML()
        assert automl.task == 'classification'
        assert automl.time_budget == 3600
        assert automl.metric == 'accuracy'
        assert automl.n_jobs == -1
        assert automl.random_state is None

    def test_init_custom_params(self):
        """Test AutoML initialization with custom parameters."""
        automl = AutoML(
            task='regression',
            time_budget=1800,
            metric='r2',
            n_jobs=2,
            random_state=42
        )
        assert automl.task == 'regression'
        assert automl.time_budget == 1800
        assert automl.metric == 'r2'
        assert automl.n_jobs == 2
        assert automl.random_state == 42

    @patch('c60.core.automl.PipelineGenerator')
    @patch('c60.core.automl.Evaluator')
    def test_fit_classification(self, mock_evaluator_class, mock_generator_class, classification_data):
        """Test fitting the AutoML model for classification."""
        X, y = classification_data
        
        # Create a mock pipeline
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.fit.return_value = mock_pipeline
        mock_pipeline.predict.return_value = np.array([0, 1, 0, 1])
        
        # Set up the generator
        mock_generator = mock_generator_class.return_value
        mock_generator.generate_initial_population.return_value = [mock_pipeline] * 3
        
        # Set up the evaluator
        mock_evaluator = mock_evaluator_class.return_value
        mock_evaluator.evaluate.return_value = {'accuracy': 0.9}
        
        # Create and fit AutoML
        automl = AutoML(task='classification', random_state=42)
        automl.fit(X, y)
        
        # Verify pipeline generator was called correctly
        mock_generator_class.assert_called_once_with(
            task='classification',
            preprocessing=True,
            feature_selection=True,
            feature_engineering=True,
            random_state=42
        )
        
        # Verify evaluator was called correctly
        mock_evaluator_class.assert_called_once_with(
            task='classification',
            metrics=['accuracy'],
            cv=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Verify fit was called on the pipeline
        mock_pipeline.fit.assert_called_once()
        
        # Verify best pipeline and score were set
        assert automl.best_pipeline_ is not None
        assert automl.best_score_ is not None

    @patch('c60.core.automl.PipelineGenerator')
    @patch('c60.core.automl.Evaluator')
    def test_fit_regression(self, mock_evaluator_class, mock_generator_class, regression_data):
        """Test fitting the AutoML model for regression."""
        X, y = regression_data
        
        # Create a mock pipeline
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.fit.return_value = mock_pipeline
        mock_pipeline.predict.return_value = np.array([0.5, 1.2, 0.8, 1.5])
        
        # Set up the generator
        mock_generator = mock_generator_class.return_value
        mock_generator.generate_initial_population.return_value = [mock_pipeline] * 3
        
        # Set up the evaluator
        mock_evaluator = mock_evaluator_class.return_value
        mock_evaluator.evaluate.return_value = {'r2': 0.85}
        
        # Create and fit AutoML
        automl = AutoML(task='regression', metric='r2', random_state=42)
        automl.fit(X, y)
        
        # Verify pipeline generator was called correctly
        mock_generator_class.assert_called_once_with(
            task='regression',
            preprocessing=True,
            feature_selection=True,
            feature_engineering=True,
            random_state=42
        )
        
        # Verify evaluator was called correctly
        mock_evaluator_class.assert_called_once_with(
            task='regression',
            metrics=['r2'],
            cv=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Verify fit was called on the pipeline
        mock_pipeline.fit.assert_called_once()
        
        # Verify best pipeline and score were set
        assert automl.best_pipeline_ is not None
        assert automl.best_score_ == 0.85

    def test_predict_not_fitted(self):
        """Test that predict raises an error when called before fit."""
        automl = AutoML()
        X = np.array([[1, 2, 3], [4, 5, 6]])
        
        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            automl.predict(X)

    @patch('c60.core.automl.PipelineGenerator')
    @patch('c60.core.automl.Evaluator')
    def test_predict(self, mock_evaluator_class, mock_generator_class, classification_data):
        """Test making predictions with a fitted AutoML model."""
        X, y = classification_data
        X_test = X[:5]  # Use first 5 samples as test data
        
        # Create a mock pipeline
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.fit.return_value = mock_pipeline
        mock_pipeline.predict.return_value = np.array([0, 1, 0, 1, 0])
        
        # Set up the generator
        mock_generator = mock_generator_class.return_value
        mock_generator.generate_initial_population.return_value = [mock_pipeline] * 3
        
        # Set up the evaluator
        mock_evaluator = mock_evaluator_class.return_value
        mock_evaluator.evaluate.return_value = {'accuracy': 0.9}
        
        # Create, fit, and predict with AutoML
        automl = AutoML(random_state=42)
        automl.fit(X, y)
        y_pred = automl.predict(X_test)
        
        # Verify predict was called on the best pipeline
        automl.best_pipeline_.predict.assert_called_once()
        
        # Verify predictions have the expected shape
        assert len(y_pred) == len(X_test)

    @patch('c60.core.automl.PipelineGenerator')
    @patch('c60.core.automl.Evaluator')
    def test_save_load(self, mock_evaluator_class, mock_generator_class, tmp_path, classification_data):
        """Test saving and loading an AutoML model."""
        X, y = classification_data
        
        # Create a mock pipeline
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.fit.return_value = mock_pipeline
        mock_pipeline.predict.return_value = np.array([0, 1, 0, 1])
        
        # Set up the generator
        mock_generator = mock_generator_class.return_value
        mock_generator.generate_initial_population.return_value = [mock_pipeline] * 3
        
        # Set up the evaluator
        mock_evaluator = mock_evaluator_class.return_value
        mock_evaluator.evaluate.return_value = {'accuracy': 0.9}
        
        # Create and fit AutoML
        automl = AutoML(random_state=42)
        automl.fit(X, y)
        
        # Save the model
        save_path = tmp_path / "automl_model.pkl"
        automl.save(save_path)
        
        # For now, just verify the method was called
        # In a real test, we would also load and verify the model
        assert save_path.exists()
