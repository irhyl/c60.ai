"""
Pipeline generation module for the C60.ai framework.

This module provides functionality for generating machine learning pipelines
using various strategies and heuristics.
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, 
    OrdinalEncoder, KBinsDiscretizer, PolynomialFeatures
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_classif, f_regression, 
    mutual_info_classif, mutual_info_regression, RFE, RFECV
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from .pipeline import Pipeline, PipelineStep


class PipelineGenerator:
    """
    A class for generating machine learning pipelines.
    
    This class provides methods for generating pipelines using various
    strategies and heuristics.
    """
    
    # Default preprocessing steps
    DEFAULT_NUMERICAL_PREPROCESSING = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
    
    DEFAULT_CATEGORICAL_PREPROCESSING = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
    
    # Model classes for different tasks
    MODEL_CLASSES = {
        'classification': {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier,
            'catboost': CatBoostClassifier
        },
        'regression': {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'ridge': Ridge,
            'lasso': Lasso,
            'elasticnet': ElasticNet,
            'svm': SVR,
            'xgboost': XGBRegressor,
            'lightgbm': LGBMRegressor,
            'catboost': CatBoostRegressor
        }
    }
    
    # Default model parameters
    DEFAULT_MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        },
        'logistic_regression': {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        },
        'svm': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True
        },
        'ridge': {
            'alpha': 1.0,
            'random_state': 42
        },
        'lasso': {
            'alpha': 1.0,
            'random_state': 42
        },
        'elasticnet': {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        },
        'lightgbm': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': -1,
            'random_state': 42
        },
        'catboost': {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_state': 42,
            'verbose': 0
        }
    }
    
    def __init__(
        self,
        task: str = 'classification',
        preprocessing: bool = True,
        feature_selection: bool = True,
        feature_engineering: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize the pipeline generator.
        
        Args:
            task: Type of task ('classification' or 'regression').
            preprocessing: Whether to include preprocessing steps.
            feature_selection: Whether to include feature selection.
            feature_engineering: Whether to include feature engineering.
            random_state: Random seed for reproducibility.
        """
        self.task = task
        self.preprocessing = preprocessing
        self.feature_selection = feature_selection
        self.feature_engineering = feature_engineering
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_baseline_pipeline(
        self,
        model_type: str = 'random_forest',
        model_params: Optional[Dict[str, Any]] = None,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Pipeline:
        """
        Generate a baseline pipeline.
        
        Args:
            model_type: Type of model to use.
            model_params: Parameters for the model.
            numerical_features: List of numerical feature names.
            categorical_features: List of categorical feature names.
            
        Returns:
            A pipeline with preprocessing and a model.
        """
        pipeline = Pipeline()
        
        # Add preprocessing steps
        if self.preprocessing:
            if numerical_features:
                pipeline.add_step(
                    name='numerical_preprocessing',
                    estimator=SklearnPipeline(self.DEFAULT_NUMERICAL_PREPROCESSING),
                    params={'memory': 'cache'}
                )
            
            if categorical_features:
                pipeline.add_step(
                    name='categorical_preprocessing',
                    estimator=SklearnPipeline(self.DEFAULT_CATEGORICAL_PREPROCESSING),
                    params={'memory': 'cache'}
                )
        
        # Add feature selection
        if self.feature_selection and (numerical_features or categorical_features):
            pipeline.add_step(
                name='feature_selector',
                estimator=SelectKBest(
                    score_func=f_classif if self.task == 'classification' else f_regression,
                    k='all'
                )
            )
        
        # Add model
        model_class = self.MODEL_CLASSES[self.task].get(model_type)
        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Merge default and user-specified parameters
        params = self.DEFAULT_MODEL_PARAMS.get(model_type, {}).copy()
        if model_params:
            params.update(model_params)
        
        pipeline.add_step(
            name='model',
            estimator=model_class(**params)
        )
        
        return pipeline
    
    def generate_advanced_pipeline(
        self,
        model_types: Optional[List[str]] = None,
        feature_engineering_steps: Optional[List[Tuple[str, Any]]] = None,
        feature_selection_methods: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Pipeline:
        """
        Generate an advanced pipeline with multiple preprocessing and modeling options.
        
        Args:
            model_types: List of model types to include in an ensemble.
            feature_engineering_steps: List of (name, transformer) tuples for feature engineering.
            feature_selection_methods: List of feature selection methods to apply.
            numerical_features: List of numerical feature names.
            categorical_features: List of categorical feature names.
            
        Returns:
            An advanced pipeline with multiple preprocessing and modeling steps.
        """
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting']
            
        if feature_selection_methods is None:
            feature_selection_methods = ['univariate', 'rfe']
        
        pipeline = Pipeline()
        
        # Add preprocessing steps
        if self.preprocessing:
            if numerical_features:
                pipeline.add_step(
                    name='numerical_preprocessing',
                    estimator=SklearnPipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('poly', PolynomialFeatures(degree=2, include_bias=False))
                    ]),
                    params={'memory': 'cache'}
                )
            
            if categorical_features:
                pipeline.add_step(
                    name='categorical_preprocessing',
                    estimator=SklearnPipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore'))
                    ]),
                    params={'memory': 'cache'}
                )
        
        # Add feature engineering
        if self.feature_engineering and feature_engineering_steps:
            for name, transformer in feature_engineering_steps:
                pipeline.add_step(
                    name=f'feature_engineering_{name}',
                    estimator=transformer
                )
        
        # Add feature selection
        if self.feature_selection and feature_selection_methods:
            for method in feature_selection_methods:
                if method == 'univariate':
                    pipeline.add_step(
                        name='univariate_feature_selection',
                        estimator=SelectKBest(
                            score_func=f_classif if self.task == 'classification' else f_regression,
                            k='all'
                        )
                    )
                elif method == 'rfe':
                    # Use a simple model for RFE
                    base_estimator = (
                        RandomForestClassifier(random_state=self.random_state) 
                        if self.task == 'classification' 
                        else RandomForestRegressor(random_state=self.random_state)
                    )
                    pipeline.add_step(
                        name='rfe_feature_selection',
                        estimator=RFE(
                            estimator=base_estimator,
                            n_features_to_select='auto',
                            step=0.1
                        )
                    )
        
        # Add models
        for i, model_type in enumerate(model_types):
            model_class = self.MODEL_CLASSES[self.task].get(model_type)
            if model_class is None:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            params = self.DEFAULT_MODEL_PARAMS.get(model_type, {}).copy()
            pipeline.add_step(
                name=f'model_{i}_{model_type}',
                estimator=model_class(**params)
            )
        
        return pipeline
    
    def generate_initial_population(
        self,
        n_pipelines: int = 10,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> List[Pipeline]:
        """
        Generate an initial population of diverse pipelines.
        
        Args:
            n_pipelines: Number of pipelines to generate.
            numerical_features: List of numerical feature names.
            categorical_features: List of categorical feature names.
            
        Returns:
            A list of diverse pipelines.
        """
        pipelines = []
        model_types = list(self.MODEL_CLASSES[self.task].keys())
        
        for i in range(n_pipelines):
            # Randomly select model type
            model_type = np.random.choice(model_types)
            
            # Randomly decide on preprocessing and feature selection
            preprocessing = self.preprocessing and np.random.rand() > 0.2  # 80% chance
            feature_selection = self.feature_selection and np.random.rand() > 0.5  # 50% chance
            
            # Generate pipeline
            pipeline = self.generate_baseline_pipeline(
                model_type=model_type,
                numerical_features=numerical_features if preprocessing else None,
                categorical_features=categorical_features if preprocessing else None
            )
            
            # Randomly modify some parameters
            for step in pipeline.steps:
                if hasattr(step.estimator, 'set_params'):
                    params = step.estimator.get_params()
                    for param in params:
                        if 'random_state' in param and self.random_state is not None:
                            step.estimator.set_params(**{param: self.random_state + i})
                        
                        # Randomly modify some hyperparameters
                        if param in ['n_estimators', 'max_iter', 'max_depth'] and np.random.rand() > 0.7:
                            if param == 'n_estimators':
                                step.estimator.set_params(**{param: np.random.randint(50, 200)})
                            elif param == 'max_iter':
                                step.estimator.set_params(**{param: np.random.randint(100, 1000)})
                            elif param == 'max_depth':
                                step.estimator.set_params(**{param: np.random.randint(3, 10)})
            
            pipelines.append(pipeline)
        
        return pipelines
