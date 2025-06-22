"""
Scoring functionality for evaluating machine learning pipelines.

This module provides the PipelineScorer class which is responsible for
evaluating the performance of machine learning pipelines using various metrics.
"""

from typing import Dict, Callable, Optional, Any, Union, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
)
from sklearn.model_selection import cross_val_score, KFold


class PipelineScorer:
    """
    A class for scoring machine learning pipelines.
    
    This class provides methods to evaluate the performance of machine learning
    pipelines using various metrics and cross-validation.
    
    Args:
        task_type (str): Type of machine learning task ('classification' or 'regression').
        metrics (Optional[Dict[str, Callable]]): Dictionary of custom metrics to use.
        cv (Optional[int]): Number of cross-validation folds. If None, no CV is performed.
        random_state (int): Random seed for reproducibility.
        calculate_feature_importance (bool): Whether to calculate feature importance.
    """
    
    def __init__(
        self,
        task_type: str = 'classification',
        metrics: Optional[Dict[str, Callable]] = None,
        cv: Optional[int] = None,
        random_state: int = 42,
        calculate_feature_importance: bool = False
    ):
        """Initialize the PipelineScorer with the given configuration."""
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")
            
        self.task_type = task_type
        self.cv = cv
        self.random_state = random_state
        self.calculate_feature_importance = calculate_feature_importance
        
        # Set default metrics based on task type
        if task_type == 'classification':
            self.metrics = {
                'accuracy': accuracy_score,
                'f1': f1_score,
                'precision': precision_score,
                'recall': recall_score,
                'roc_auc': roc_auc_score
            }
        else:  # regression
            self.metrics = {
                'r2': r2_score,
                'mse': mean_squared_error,
                'mae': mean_absolute_error,
                'explained_variance': explained_variance_score
            }
            
        # Add any custom metrics
        if metrics:
            self.metrics.update(metrics)
    
    def score(
        self,
        pipeline: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Score the given pipeline on the provided data.
        
        Args:
            pipeline: A scikit-learn compatible pipeline or estimator.
            X: Feature matrix.
            y: Target vector.
            
        Returns:
            Dictionary mapping metric names to their values.
        """
        scores = {}
        
        # Make predictions
        if hasattr(pipeline, 'predict_proba') and self.task_type == 'classification':
            y_pred_proba = pipeline.predict_proba(X)
            if y_pred_proba.shape[1] == 2:  # binary classification
                y_pred = pipeline.predict(X)
                y_score = y_pred_proba[:, 1]
            else:  # multiclass
                y_pred = y_pred_proba.argmax(axis=1)
                y_score = y_pred_proba
        else:
            y_pred = pipeline.predict(X)
            y_score = y_pred
        
        # Calculate metrics
        for metric_name, metric_func in self.metrics.items():
            try:
                if metric_name == 'roc_auc':
                    # Handle roc_auc which needs probability estimates
                    if hasattr(pipeline, 'predict_proba'):
                        scores[metric_name] = metric_func(y, y_score)
                else:
                    scores[metric_name] = metric_func(y, y_pred)
            except Exception as e:
                print(f"Warning: Could not calculate {metric_name}: {str(e)}")
        
        # Perform cross-validation if requested
        if self.cv is not None and self.cv > 1:
            cv_scores = self._cross_validate(pipeline, X, y)
            scores.update(cv_scores)
        
        # Calculate feature importance if requested and available
        if self.calculate_feature_importance and hasattr(pipeline, 'feature_importances_'):
            scores['feature_importances'] = pipeline.feature_importances_.tolist()
        
        return scores
    
    def _cross_validate(
        self,
        pipeline: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Perform cross-validation and return the mean scores.
        
        Args:
            pipeline: A scikit-learn compatible pipeline or estimator.
            X: Feature matrix.
            y: Target vector.
            
        Returns:
            Dictionary mapping cv_metric names to their mean values.
        """
        cv_scores = {}
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Use the first metric as the scoring metric for cross-validation
        scoring_metric = next(iter(self.metrics.keys()))
        
        # Get the scoring function for cross-validation
        if scoring_metric == 'roc_auc' and hasattr(pipeline, 'predict_proba'):
            # For roc_auc, we need to use predict_proba
            scorer = 'roc_auc'
        else:
            # For other metrics, use the metric function directly
            scorer = self.metrics[scoring_metric]
        
        # Perform cross-validation
        cv_results = cross_val_score(
            pipeline, X, y, 
            cv=kf, 
            scoring=scorer,
            n_jobs=-1
        )
        
        # Store the cross-validation results
        cv_scores[f'cv_{scoring_metric}'] = np.mean(cv_results)
        cv_scores[f'cv_{scoring_metric}_std'] = np.std(cv_results)
        
        return cv_scores
