"""
Tests for the optimizer module.

This module contains unit tests for the pipeline optimization functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, call
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from c60.core.optimizer import PipelineOptimizer


class TestPipelineOptimizer:
    """Test suite for the PipelineOptimizer class."""

    @pytest.fixture
    def classification_data(self):
        """Create sample classification data for testing."""
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
        """Create sample regression data for testing."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=5,
            noise=0.1,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        pipeline = MagicMock()
        pipeline.get_params.return_value = {'param1': 1, 'param2': 2}
        pipeline.set_params.return_value = pipeline
        return pipeline

    @pytest.fixture
    def mock_study(self):
        """Create a mock Optuna study."""
        study = MagicMock()
        study.best_params = {'param1': 10, 'param2': 20}
        study.best_value = 0.95
        return study

    @pytest.fixture
    def mock_trial(self):
        """Create a mock Optuna trial."""
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5
        trial.suggest_int.return_value = 10
        trial.suggest_categorical.return_value = 'auto'
        return trial

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        optimizer = PipelineOptimizer()
        assert optimizer.n_trials == 100
        assert optimizer.timeout is None
        assert optimizer.study_name is None
        assert optimizer.storage is None
        assert optimizer.load_if_exists is False
        assert optimizer.direction == 'maximize'
        assert optimizer.random_state is None
        assert optimizer.n_jobs == 1

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        optimizer = PipelineOptimizer(
            n_trials=50,
            timeout=3600,
            study_name='test_study',
            storage='sqlite:///test.db',
            load_if_exists=True,
            direction='minimize',
            random_state=42,
            n_jobs=2
        )
        assert optimizer.n_trials == 50
        assert optimizer.timeout == 3600
        assert optimizer.study_name == 'test_study'
        assert optimizer.storage == 'sqlite:///test.db'
        assert optimizer.load_if_exists is True
        assert optimizer.direction == 'minimize'
        assert optimizer.random_state == 42
        assert optimizer.n_jobs == 2

    @patch('optuna.create_study')
    @patch('optuna.study.Study.optimize')
    def test_optimize_hyperparameters_classification(
        self, mock_optimize, mock_create_study, mock_study, classification_data, mock_pipeline
    ):
        """Test hyperparameter optimization for classification."""
        X, y = classification_data
        mock_create_study.return_value = mock_study
        
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        
        optimizer = PipelineOptimizer(n_trials=10, random_state=42)
        best_params, best_score = optimizer.optimize_hyperparameters(
            mock_pipeline, X, y, param_grid, scoring='f1_macro', cv=5
        )
        
        # Verify study was created with correct parameters
        mock_create_study.assert_called_once_with(
            direction='maximize',
            study_name=None,
            storage=None,
            load_if_exists=False,
            sampler=None,
            pruner=None,
        )
        
        # Verify optimize was called with correct parameters
        mock_optimize.assert_called_once()
        
        # Verify best parameters and score were returned
        assert best_params == mock_study.best_params
        assert best_score == mock_study.best_value

    @patch('optuna.create_study')
    @patch('optuna.study.Study.optimize')
    def test_optimize_hyperparameters_regression(
        self, mock_optimize, mock_create_study, mock_study, regression_data, mock_pipeline
    ):
        """Test hyperparameter optimization for regression."""
        X, y = regression_data
        mock_create_study.return_value = mock_study
        
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        
        optimizer = PipelineOptimizer(n_trials=10, direction='minimize', random_state=42)
        best_params, best_score = optimizer.optimize_hyperparameters(
            mock_pipeline, X, y, param_grid, scoring='neg_mean_squared_error', cv=5
        )
        
        # Verify study was created with correct parameters
        mock_create_study.assert_called_once_with(
            direction='minimize',
            study_name=None,
            storage=None,
            load_if_exists=False,
            sampler=None,
            pruner=None,
        )
        
        # Verify optimize was called with correct parameters
        mock_optimize.assert_called_once()
        
        # Verify best parameters and score were returned
        assert best_params == mock_study.best_params
        assert best_score == mock_study.best_value

    @patch('optuna.create_study')
    @patch('optuna.study.Study.optimize')
    def test_optimize_with_custom_study_params(
        self, mock_optimize, mock_create_study, mock_study, classification_data, mock_pipeline
    ):
        """Test optimization with custom study parameters."""
        X, y = classification_data
        mock_create_study.return_value = mock_study
        
        param_grid = {'param1': [1, 10], 'param2': [1, 100]}
        
        optimizer = PipelineOptimizer(
            n_trials=20,
            study_name='test_study',
            storage='sqlite:///test.db',
            load_if_exists=True,
            direction='maximize',
            random_state=42
        )
        
        best_params, best_score = optimizer.optimize_hyperparameters(
            mock_pipeline, X, y, param_grid, scoring='accuracy', cv=3
        )
        
        # Verify study was created with custom parameters
        mock_create_study.assert_called_once_with(
            direction='maximize',
            study_name='test_study',
            storage='sqlite:///test.db',
            load_if_exists=True,
            sampler=None,
            pruner=None,
        )
        
        # Verify optimize was called with correct parameters
        mock_optimize.assert_called_once()
        
        # Verify best parameters and score were returned
        assert best_params == mock_study.best_params
        assert best_score == mock_study.best_value

    @patch('optuna.create_study')
    @patch('optuna.study.Study.optimize')
    def test_optimize_with_timeout(
        self, mock_optimize, mock_create_study, mock_study, classification_data, mock_pipeline
    ):
        """Test optimization with a timeout."""
        X, y = classification_data
        mock_create_study.return_value = mock_study
        
        param_grid = {'param1': [1, 10], 'param2': [1, 100]}
        
        optimizer = PipelineOptimizer(n_trials=100, timeout=60)
        best_params, best_score = optimizer.optimize_hyperparameters(
            mock_pipeline, X, y, param_grid, scoring='accuracy', cv=3
        )
        
        # Verify optimize was called with timeout
        mock_optimize.assert_called_once()
        call_args = mock_optimize.call_args[1]
        assert 'timeout' in call_args
        assert call_args['timeout'] == 60

    def test_suggest_parameters(self, mock_trial):
        """Test parameter suggestion from trial."""
        param_grid = {
            'int_param': [1, 2, 3],
            'float_param': [0.1, 0.5, 1.0],
            'categorical_param': ['a', 'b', 'c'],
            'nested.param': [10, 20, 30]
        }
        
        optimizer = PipelineOptimizer()
        params = optimizer._suggest_parameters(mock_trial, param_grid)
        
        # Verify correct suggest methods were called
        expected_calls = [
            call('int_param', 1, 3, step=1),
            call('float_param', 0.1, 1.0, step=None, log=False),
            call('categorical_param', ['a', 'b', 'c']),
            call('nested.param', 10, 30, step=1)
        ]
        
        # Verify suggest_int was called for integer parameters
        assert mock_trial.suggest_int.call_count == 2  # int_param and nested.param
        assert mock_trial.suggest_float.call_count == 1  # float_param
        assert mock_trial.suggest_categorical.call_count == 1  # categorical_param
        
        # Verify all parameters are in the result
        assert set(params.keys()) == {
            'int_param', 'float_param', 'categorical_param', 'nested.param'
        }
        
        # Verify the results
        assert best_params == {'param1': 10, 'param2': 20}
        assert best_score == 0.95
        mock_pipeline.set_params.assert_called_once_with(param1=10, param2=20)

    def test_feature_selection(self, sample_data):
        """Test feature selection optimization."""
        X, y = sample_data
        
        # Add a noisy feature
        X['noise'] = np.random.rand(100)
        
        optimizer = PipelineOptimizer()
        selected_features = optimizer.optimize_feature_selection(
            X, y, n_features=2, task_type='classification'
        )
        
        # Should return 2 feature names
        assert len(selected_features) == 2
        assert all(feat in X.columns for feat in selected_features)
        assert 'noise' not in selected_features  # Should be filtered out

    @patch('engine.optimizer.PipelineOptimizer.optimize_hyperparameters')
    def test_optimize_pipeline(self, mock_optimize, sample_data, mock_pipeline):
        """Test full pipeline optimization."""
        X, y = sample_data
        
        # Mock the hyperparameter optimization
        mock_optimize.return_value = ({'param1': 10}, 0.95)
        
        optimizer = PipelineOptimizer()
        result = optimizer.optimize_pipeline(
            mock_pipeline, X, y, 
            param_grid={'param1': [1, 10]},
            feature_selection=True,
            n_features=2
        )
        
        # Verify the result structure
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'selected_features' in result
        assert result['best_score'] == 0.95
        
        # Verify the mock was called
        mock_optimize.assert_called_once()

    def test_custom_objective_function(self, sample_data, mock_pipeline):
        """Test optimization with a custom objective function."""
        X, y = sample_data
        
        def custom_objective(trial):
            return 0.9  # Dummy score
            
        optimizer = PipelineOptimizer()
        best_params, best_score = optimizer.optimize_hyperparameters(
            mock_pipeline, X, y, 
            param_grid={'param1': [1, 10]},
            objective=custom_objective
        )
        
        assert best_score == 0.9

    def test_early_stopping(self, sample_data, mock_pipeline):
        """Test early stopping during optimization."""
        X, y = sample_data
        
        # Patch Optuna's study to simulate early stopping
        with patch('optuna.create_study') as mock_study:
            study = MagicMock()
            study.best_params = {'param1': 5}
            study.best_value = 0.9
            mock_study.return_value = study
            
            # Make the study raise the exception on the second call
            study.optimize.side_effect = [None, optuna.TrialPruned()]
            
            optimizer = PipelineOptimizer(n_trials=10, early_stopping_rounds=2)
            best_params, best_score = optimizer.optimize_hyperparameters(
                mock_pipeline, X, y, {'param1': [1, 10]}
            )
        
        # Should return the best parameters found before pruning
        assert best_params == {'param1': 5}
        assert best_score == 0.9
