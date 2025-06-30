"""
Evaluation module for the C60.ai framework.

This module provides functionality for evaluating machine learning models
and pipelines with various metrics and cross-validation strategies.
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, make_scorer
)
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold


class Evaluator:
    """
    A class for evaluating machine learning models and pipelines.

    Provides methods for evaluating models using various metrics and cross-validation strategies.
    """

    # Default metrics for different task types
    DEFAULT_METRICS = {
        'classification': ['accuracy', 'f1_weighted', 'roc_auc_ovr'],
        'regression': ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']
    }
    
    # Mapping of metric names to scoring functions
    METRIC_FUNCTIONS = {
        # Classification metrics
        'accuracy': accuracy_score,
        'f1': f1_score,
        'f1_weighted': lambda y_true, y_pred, **kwargs: f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score,
        'recall': recall_score,
        'roc_auc': roc_auc_score,
        'roc_auc_ovr': lambda y_true, y_pred, **kwargs: roc_auc_score(y_true, y_pred, multi_class='ovr'),
        
        # Regression metrics
        'mse': mean_squared_error,
        'rmse': lambda y_true, y_pred, **kwargs: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error,
        'r2': r2_score,
        'neg_mean_squared_error': lambda y_true, y_pred, **kwargs: -mean_squared_error(y_true, y_pred),
        'neg_mean_absolute_error': lambda y_true, y_pred, **kwargs: -mean_absolute_error(y_true, y_pred),
    }
    
    def __init__(
        self,
        task: str = 'classification',
        metrics: Optional[List[str]] = None,
        cv: int = 5,
        random_state: Optional[int] = None,
        n_jobs: int = -1
    ):
        """
        Initialize the evaluator.
        
        Args:
            task: Type of task ('classification' or 'regression').
            metrics: List of metric names to use. If None, uses default metrics.
            cv: Number of cross-validation folds.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs to run.
        """
        self.task = task
        self.metrics = metrics or self.DEFAULT_METRICS.get(task, [])
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    def get_scorer(self, metric: str) -> Callable:
        """
        Get a scoring function for the given metric.
        
        Args:
            metric: Name of the metric.
            
        Returns:
            Scoring function.
            
        Raises:
            ValueError: If the metric is not supported.
        """
        if metric not in self.METRIC_FUNCTIONS:
            raise ValueError(f"Unsupported metric: {metric}")
        return self.METRIC_FUNCTIONS[metric]
    
    def evaluate(
        self,
        estimator: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate a model on the given data.
        
        Args:
            estimator: The model or pipeline to evaluate.
            X: Training features.
            y: Training target values.
            X_val: Optional validation features. If not provided, uses cross-validation.
            y_val: Optional validation target values.
            sample_weight: Optional sample weights.
            
        Returns:
            Dictionary mapping metric names to their values.
        """
        if X_val is not None and y_val is not None:
            return self._evaluate_holdout(estimator, X, y, X_val, y_val, sample_weight)
        return self._evaluate_cv(estimator, X, y, sample_weight)
    
    def _evaluate_holdout(
        self,
        estimator: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate a model on a holdout set.
        
        Args:
            estimator: The model or pipeline to evaluate.
            X_train: Training features.
            y_train: Training target values.
            X_val: Validation features.
            y_val: Validation target values.
            sample_weight: Optional sample weights.
            
        Returns:
            Dictionary mapping metric names to their values.
        """
        # Fit the model
        estimator.fit(X_train, y_train)
        
        # Make predictions
        if hasattr(estimator, 'predict_proba') and self.task == 'classification':
            y_pred_proba = estimator.predict_proba(X_val)
            if y_pred_proba.shape[1] == 2:  # binary classification
                y_pred = estimator.predict(X_val)
                y_score = y_pred_proba[:, 1]
            else:  # multiclass
                y_pred = y_pred_proba.argmax(axis=1)
                y_score = y_pred_proba
        else:
            y_pred = estimator.predict(X_val)
            y_score = y_pred
        
        # Calculate metrics
        results = {}
        for metric in self.metrics:
            try:
                if metric == 'roc_auc' or metric.startswith('roc_auc_'):
                    # Handle roc_auc which needs probability estimates
                    if hasattr(estimator, 'predict_proba'):
                        results[metric] = self.get_scorer(metric)(y_val, y_score)
                else:
                    results[metric] = self.get_scorer(metric)(y_val, y_pred)
            except Exception as e:
                print(f"Warning: Could not calculate {metric}: {str(e)}")
                results[metric] = float('nan')
        
        return results
    
    def _evaluate_cv(
        self,
        estimator: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate a model using cross-validation.
        
        Args:
            estimator: The model or pipeline to evaluate.
            X: Features.
            y: Target values.
            sample_weight: Optional sample weights.
            
        Returns:
            Dictionary mapping metric names to their mean values across folds.
        """
        # Create cross-validation strategy
        if self.task == 'classification':
            cv = StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            cv = KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state
            )
        
        # Create scorers for all metrics
        scorers = {}
        for metric in self.metrics:
            if metric in self.METRIC_FUNCTIONS:
                scorers[metric] = make_scorer(
                    self.METRIC_FUNCTIONS[metric],
                    needs_proba=metric.startswith('roc_auc')
                )
        
        # Perform cross-validation
        cv_results = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring=scorers,
            cv=cv,
            n_jobs=self.n_jobs,
            return_train_score=False,
            error_score='raise'
        )
        
        # Calculate mean scores
        results = {}
        for metric in self.metrics:
            test_metric = f'test_{metric}'
            if test_metric in cv_results:
                results[metric] = np.mean(cv_results[test_metric])
                results[f'{metric}_std'] = np.std(cv_results[test_metric])
        
        return results
