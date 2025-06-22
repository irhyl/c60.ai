"""
Optimization functionality for machine learning pipelines.

This module provides the PipelineOptimizer class which is responsible for
optimizing machine learning pipelines using techniques like hyperparameter
tuning and feature selection.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import get_scorer


class PipelineOptimizer:
    """
    A class for optimizing machine learning pipelines.
    
    This class provides methods to optimize machine learning pipelines using
    techniques like hyperparameter tuning and feature selection.
    
    Args:
        n_trials (int): Number of optimization trials.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric to optimize.
        direction (str): Direction of optimization ('maximize' or 'minimize').
        random_state (int): Random seed for reproducibility.
        early_stopping_rounds (Optional[int]): Number of rounds for early stopping.
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        cv: int = 5,
        scoring: str = 'accuracy',
        direction: str = 'maximize',
        random_state: int = 42,
        early_stopping_rounds: Optional[int] = None
    ):
        """Initialize the PipelineOptimizer with the given configuration."""
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.direction = direction
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        
        if direction not in ['minimize', 'maximize']:
            raise ValueError("direction must be either 'minimize' or 'maximize'")
    
    def optimize_hyperparameters(
        self,
        pipeline: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Dict[str, List[Any]],
        objective: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            pipeline: A scikit-learn compatible pipeline or estimator.
            X: Feature matrix.
            y: Target vector.
            param_grid: Dictionary mapping parameter names to lists of values.
            objective: Custom objective function. If None, uses the default.
            
        Returns:
            Tuple containing the best parameters and the best score.
        """
        # Create a study
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Create the objective function if not provided
        if objective is None:
            def objective(trial):
                params = {}
                for param_name, param_values in param_grid.items():
                    if all(isinstance(x, (int, float)) for x in param_values):
                        # Continuous parameter
                        params[param_name] = trial.suggest_float(
                            param_name, min(param_values), max(param_values), log=True
                        )
                    elif all(isinstance(x, bool) for x in param_values):
                        # Boolean parameter
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    elif all(isinstance(x, str) for x in param_values):
                        # Categorical parameter
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        # Mixed types, use categorical
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                
                # Set parameters
                pipeline.set_params(**params)
                
                # Calculate score with cross-validation
                kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                score = cross_val_score(
                    pipeline, X, y, 
                    cv=kf, 
                    scoring=self.scoring,
                    n_jobs=-1
                ).mean()
                
                return score
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=None,
            show_progress_bar=True
        )
        
        # Get best parameters and score
        best_params = study.best_params
        best_score = study.best_value
        
        # Set the best parameters on the pipeline
        pipeline.set_params(**best_params)
        
        return best_params, best_score
    
    def optimize_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        task_type: str = 'classification',
        method: str = 'f_classif'
    ) -> List[str]:
        """
        Select the best features using statistical tests.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            n_features: Number of features to select.
            task_type: Type of task ('classification' or 'regression').
            method: Feature selection method.
            
        Returns:
            List of selected feature names.
        """
        if task_type == 'classification':
            if method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=n_features)
            elif method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            else:
                raise ValueError(f"Unsupported feature selection method: {method}")
        else:  # regression
            if method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=n_features)
            elif method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            else:
                raise ValueError(f"Unsupported feature selection method: {method}")
        
        # Fit selector
        selector.fit(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
        # Return selected feature names
        if isinstance(X, pd.DataFrame):
            return X.columns[selected_indices].tolist()
        return [f"feature_{i}" for i in selected_indices]
    
    def optimize_pipeline(
        self,
        pipeline: Any,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        feature_selection: bool = False,
        n_features: Optional[int] = None,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Optimize the entire pipeline including feature selection.
        
        Args:
            pipeline: A scikit-learn compatible pipeline or estimator.
            X: Feature matrix.
            y: Target vector.
            param_grid: Dictionary mapping parameter names to lists of values.
            feature_selection: Whether to perform feature selection.
            n_features: Number of features to select.
            task_type: Type of task ('classification' or 'regression').
            
        Returns:
            Dictionary containing optimization results.
        """
        result = {}
        
        # Perform feature selection if requested
        if feature_selection and n_features is not None:
            selected_features = self.optimize_feature_selection(X, y, n_features, task_type)
            X = X[selected_features]
            result['selected_features'] = selected_features
        
        # Optimize hyperparameters
        best_params, best_score = self.optimize_hyperparameters(
            pipeline, X, y, param_grid
        )
        
        result.update({
            'best_params': best_params,
            'best_score': best_score,
            'feature_selection_performed': feature_selection,
            'n_features_selected': n_features if feature_selection else X.shape[1]
        })
        
        return result
