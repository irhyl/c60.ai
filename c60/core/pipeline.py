"""
Pipeline module for the C60.ai framework.

Defines the Pipeline and PipelineStep classes for constructing and managing
machine learning pipelines in C60.ai.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline as SklearnPipeline
from ..introspect import HybridNode


class PipelineStep:
    """
    A single step in a machine learning pipeline.
    Supports hybrid symbolic/neural nodes via HybridNode.
    """
    def __init__(
        self,
        name: str,
        estimator: Union[BaseEstimator, TransformerMixin, HybridNode],
        params: Optional[Dict[str, Any]] = None,
        is_frozen: bool = False
    ):
        self.name = name
        self.is_frozen = is_frozen
        self.params = params or {}
        if isinstance(estimator, HybridNode):
            self.estimator = estimator
        else:
            self.estimator = clone(estimator)
            if self.params:
                self.estimator.set_params(**self.params)

    def is_hybrid(self) -> bool:
        """Return True if this step is a HybridNode (symbolic+neural)."""
        return isinstance(self.estimator, HybridNode)

    def mutate(self):
        """Mutate this step if hybrid (demo: swap symbolic/neural)."""
        if self.is_hybrid():
            if self.estimator.is_symbolic() and self.estimator.is_neural():
                self.estimator.symbolic_rule, self.estimator.neural_model = (
                    self.estimator.neural_model, self.estimator.symbolic_rule)
        return self
    
    def set_params(self, **params) -> 'PipelineStep':
        """
        Set parameters for this step.
        
        Args:
            **params: Parameters to set.
            
        Returns:
            self: Returns the instance itself.
        """
        if not self.is_frozen:
            self.estimator.set_params(**params)
        return self
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this step.
        
        Args:
            deep: If True, will return the parameters for this step and
                  contained subobjects.
                  
        Returns:
            Parameter names mapped to their values.
        """
        return self.estimator.get_params(deep=deep)
    
    def __repr__(self) -> str:
        """String representation of the pipeline step."""
        return f"{self.__class__.__name__}(name='{self.name}', estimator={self.estimator.__class__.__name__})"


class Pipeline:
    """
    Pipeline supporting hybrid symbolic and neural nodes.
    Allows mutation/crossover between symbolic and neural steps.
    """
    def __init__(self, steps: Optional[List[PipelineStep]] = None):
        self.steps = steps or []
        self._sklearn_pipeline = None
        self._build_sklearn_pipeline()
        self.metadata = {}

    def mutate(self):
        """Mutate pipeline: hybrid-aware mutation (mutates hybrid steps)."""
        for step in self.steps:
            if hasattr(step, 'mutate'):
                step.mutate()
        return self

    def crossover(self, other: 'Pipeline') -> 'Pipeline':
        """Crossover with another pipeline (swap half steps)."""
        cut = len(self.steps) // 2
        new_steps = self.steps[:cut] + other.steps[cut:]
        return Pipeline(steps=new_steps)
    
    def _build_sklearn_pipeline(self) -> None:
        """Build the underlying scikit-learn pipeline."""
        sklearn_steps = [(step.name, step.estimator) for step in self.steps]
        self._sklearn_pipeline = SklearnPipeline(sklearn_steps)
    
    def add_step(
        self,
        name: str,
        estimator: Union[BaseEstimator, TransformerMixin],
        params: Optional[Dict[str, Any]] = None,
        index: Optional[int] = None
    ) -> 'Pipeline':
        """
        Add a step to the pipeline.
        
        Args:
            name: Name of the step.
            estimator: The estimator or transformer to add.
            params: Parameters for the estimator.
            index: Position to insert the step at. If None, appends to the end.
            
        Returns:
            self: Returns the instance itself.
        """
        step = PipelineStep(name=name, estimator=estimator, params=params)
        if index is None:
            self.steps.append(step)
        else:
            self.steps.insert(index, step)
        self._build_sklearn_pipeline()
        return self
    
    def remove_step(self, name: str) -> 'Pipeline':
        """
        Remove a step from the pipeline.
        
        Args:
            name: Name of the step to remove.
            
        Returns:
            self: Returns the instance itself.
        """
        self.steps = [step for step in self.steps if step.name != name]
        self._build_sklearn_pipeline()
        return self
    
    def get_step(self, name: str) -> Optional[PipelineStep]:
        """
        Get a step by name.
        
        Args:
            name: Name of the step to get.
            
        Returns:
            The requested step, or None if not found.
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def set_step_params(self, name: str, **params) -> 'Pipeline':
        """
        Set parameters for a step.
        
        Args:
            name: Name of the step.
            **params: Parameters to set.
            
        Returns:
            self: Returns the instance itself.
        """
        step = self.get_step(name)
        if step:
            step.set_params(**params)
            self._build_sklearn_pipeline()
        return self
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None, **fit_params):
        """
        Fit the pipeline on the given data.
        
        Args:
            X: Training data.
            y: Target values.
            **fit_params: Additional fitting parameters.
            
        Returns:
            self: Returns the instance itself.
        """
        self._sklearn_pipeline.fit(X, y, **fit_params)
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Apply transforms and predict with the final estimator.
        
        Args:
            X: Input data.
            
        Returns:
            Predicted values.
        """
        return self._sklearn_pipeline.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input data.
            
        Returns:
            Class probabilities.
        """
        return self._sklearn_pipeline.predict_proba(X)
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Apply transforms to the data.
        
        Args:
            X: Input data.
            
        Returns:
            Transformed data.
        """
        return self._sklearn_pipeline.transform(X)
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> float:
        """
        Score the pipeline on the given data.
        
        Args:
            X: Input data.
            y: True values.
            
        Returns:
            Score of the pipeline.
        """
        return self._sklearn_pipeline.score(X, y)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this pipeline.
        
        Args:
            deep: If True, will return the parameters for this pipeline and
                  contained subobjects.
                  
        Returns:
            Parameter names mapped to their values.
        """
        params = {}
        for step in self.steps:
            step_params = step.get_params(deep=deep)
            for param_name, param_value in step_params.items():
                params[f"{step.name}__{param_name}"] = param_value
        return params
    
    def set_params(self, **params) -> 'Pipeline':
        """
        Set the parameters of this pipeline.
        
        Args:
            **params: Parameters to set.
            
        Returns:
            self: Returns the instance itself.
        """
        # Group parameters by step
        step_params = {}
        for param_name, param_value in params.items():
            if "__" in param_name:
                step_name, param = param_name.split("__", 1)
                if step_name not in step_params:
                    step_params[step_name] = {}
                step_params[step_name][param] = param_value
        
        # Set parameters for each step
        for step_name, params in step_params.items():
            step = self.get_step(step_name)
            if step is not None:
                step.set_params(**params)
        
        self._build_sklearn_pipeline()
        return self
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return f"{self.__class__.__name__}(steps={self.steps})"
