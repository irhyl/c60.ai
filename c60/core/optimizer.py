"""
Optimizer module for the C60.ai framework.

Defines the Optimizer class for hyperparameter and feature selection optimization
using Optuna and other strategies in C60.ai.
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from .evaluator import Evaluator
from .pipeline import Pipeline


class Optimizer:
    """
    A class for optimizing machine learning models and pipelines.
    
    This class provides methods for hyperparameter optimization using Optuna.
    """
    
    def __init__(
        self,
        task: str = 'classification',
        metric: str = None,
        direction: str = 'maximize',
        n_trials: int = 100,
        timeout: Optional[float] = None,
        cv: int = 5,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        sampler: str = 'tpe',
        pruner: str = 'hyperband',
        **kwargs
    ):
        """
        Initialize the optimizer.
        
        Args:
            task: Type of task ('classification' or 'regression').
            metric: Metric to optimize.
            direction: Direction of optimization ('minimize' or 'maximize').
            n_trials: Maximum number of optimization trials.
            timeout: Maximum time in seconds for optimization.
            cv: Number of cross-validation folds.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs to run.
            sampler: Optimization sampler ('tpe', 'random', 'cmaes', etc.).
            pruner: Pruning strategy ('hyperband', 'median', etc.).
            **kwargs: Additional arguments for the optimizer.
        """
        self.task = task
        self.metric = metric or ('accuracy' if task == 'classification' else 'neg_mean_squared_error')
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.sampler_name = sampler
        self.pruner_name = pruner
        self.kwargs = kwargs
        
        # Initialize evaluator
        self.evaluator = Evaluator(
            task=task,
            metrics=[self.metric],
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        # Initialize sampler
        if self.sampler_name == 'tpe':
            self.sampler = TPESampler(seed=random_state)
        elif self.sampler_name == 'random':
            self.sampler = optuna.samplers.RandomSampler(seed=random_state)
        elif self.sampler_name == 'cmaes':
            self.sampler = optuna.samplers.CmaEsSampler(seed=random_state)
        else:
            self.sampler = optuna.samplers.TPESampler(seed=random_state)
        
        # Initialize pruner
        if self.pruner_name == 'hyperband':
            self.pruner = HyperbandPruner()
        elif self.pruner_name == 'median':
            self.pruner = optuna.pruners.MedianPruner()
        else:
            self.pruner = None
    
    def optimize(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_distributions: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            estimator: The model or pipeline to optimize.
            X: Training features.
            y: Training target values.
            param_distributions: Dictionary with parameters names as keys and distributions
                               or lists of parameters to try.
            **kwargs: Additional arguments for the optimization.
            
        Returns:
            Dictionary with the best parameters and the optimization study.
        """
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=f"{estimator.__class__.__name__}_optimization"
        )
        
        # Define objective function
        def objective(trial):
            # Suggest parameters
            params = {}
            for param_name, param_dist in param_distributions.items():
                if isinstance(param_dist, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_dist)
                elif isinstance(param_dist, tuple) and len(param_dist) == 3 and param_dist[0] == 'int':
                    # Integer parameter with range and step
                    params[param_name] = trial.suggest_int(param_name, param_dist[1], param_dist[2])
                elif isinstance(param_dist, tuple) and len(param_dist) == 2 and all(isinstance(x, (int, float)) for x in param_dist):
                    # Float parameter with range
                    params[param_name] = trial.suggest_float(param_name, param_dist[0], param_dist[1])
                elif isinstance(param_dist, dict) and 'type' in param_dist:
                    # Complex parameter distribution
                    if param_dist['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, param_dist['values'])
                    elif param_dist['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_dist.get('low', 0),
                            param_dist.get('high', 1000),
                            step=param_dist.get('step', 1)
                        )
                    elif param_dist['type'] == 'float':
                        if 'log' in param_dist and param_dist['log']:
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_dist.get('low', 1e-10),
                                param_dist.get('high', 1.0),
                                log=True
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_dist.get('low', 0.0),
                                param_dist.get('high', 1.0)
                            )
            
            # Set parameters
            estimator.set_params(**params)
            
            # Evaluate model
            scores = self.evaluator.evaluate(estimator, X, y)
            
            # Return the score for the optimization metric
            return scores[self.metric]
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            **kwargs
        )
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def optimize_pipeline(
        self,
        pipeline: Pipeline,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_distributions: Dict[str, Any],
        feature_selection: bool = False,
        n_features: Optional[int] = None,
        task_type: str = 'classification',
        feature_selection_method: str = 'f_classif',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize the entire pipeline including optional feature selection.
        
        Args:
            pipeline: The pipeline to optimize.
            X: Training features.
            y: Training target values.
            param_distributions: Dictionary with parameters names as keys and distributions
                               or lists of parameters to try.
            feature_selection: Whether to perform feature selection.
            n_features: Number of features to select.
            task_type: Type of task ('classification' or 'regression').
            feature_selection_method: Feature selection method ('f_classif', 'mutual_info', etc.).
            **kwargs: Additional arguments for the optimization.
        Returns:
            Dictionary with the best parameters, the optimization study, and selected features if applicable.
        """
        result = {}
        X_selected = X
        # Perform feature selection if requested
        if feature_selection and n_features is not None:
            selected_features = self.optimize_feature_selection(
                X, y, n_features, task_type, feature_selection_method
            )
            if isinstance(X, pd.DataFrame):
                X_selected = X[selected_features]
            result['selected_features'] = selected_features
        # Convert pipeline parameters to Optuna format
        optuna_params = {}
        for param_name, param_dist in param_distributions.items():
            # Add step name to parameter name
            if '__' in param_name:
                step_name, param = param_name.split('__', 1)
                optuna_params[param_name] = param_dist
            else:
                # If no step specified, apply to all steps that have this parameter
                for step in pipeline.steps:
                    if hasattr(step.estimator, param_name):
                        optuna_params[f"{step.name}__{param_name}"] = param_dist
        # Run optimization
        opt_result = self.optimize(pipeline, X_selected, y, optuna_params, **kwargs)
        result.update(opt_result)
        result['feature_selection_performed'] = feature_selection
        result['n_features_selected'] = n_features if feature_selection else (X.shape[1] if hasattr(X, 'shape') else None)
        return result

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
        from sklearn.feature_selection import (
            SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
        )
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
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        if isinstance(X, pd.DataFrame):
            return X.columns[selected_indices].tolist()
        return [f"feature_{i}" for i in selected_indices]
