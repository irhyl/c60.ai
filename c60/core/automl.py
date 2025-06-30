"""
Main AutoML class for the C60.ai framework.

Defines the AutoML class, the main entry point for automated machine learning
pipeline search, training, and model management in C60.ai.
"""

from typing import Optional, Union, Dict, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.config import Config
from ..utils.logging import get_logger
from .pipeline import Pipeline
from .evaluator import Evaluator
from .optimizer import Optimizer
from .generator import PipelineGenerator
from c60.introspect.introspector import PipelineIntrospector
from c60.introspect.agents import RLSearchAgent, NASearchAgent
from c60.introspect.story import PipelineStory


class AutoML:
    """
    Main AutoML class for automated machine learning pipeline construction.
    
    This class orchestrates the entire AutoML process including data preprocessing,
    feature engineering, model selection, hyperparameter optimization, and evaluation.
    
    Args:
        task: Type of machine learning task ('classification' or 'regression').
        time_budget: Maximum time in seconds for the AutoML process.
        metric: Evaluation metric to optimize.
        n_jobs: Number of parallel jobs to run.
        random_state: Random seed for reproducibility.
        config: Optional configuration dictionary.
    """
    
    def __init__(
        self,
        task: str = 'classification',
        time_budget: int = 3600,
        metric: Optional[str] = None,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.task = task
        self.time_budget = time_budget
        self.metric = metric or self._get_default_metric()
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.config = Config()
        if config:
            self.config.update(config)
        # Initialize components
        self.logger = get_logger(__name__)
        self.evaluator = Evaluator(task=task, metric=self.metric)
        self.optimizer = Optimizer(
            task=task,
            metric=self.metric,
            n_jobs=n_jobs,
            random_state=random_state
        )
        self.pipeline_generator = PipelineGenerator(
            task=task,
            random_state=random_state
        )
        # Introspector for explainability
        self.introspector = PipelineIntrospector()
        # State
        self.best_pipeline_ = None
        self.best_score_ = None
        self.history_ = []

    def explain_pipeline(self, pipeline_id: str) -> str:
        """Return a human-readable explanation/story for a pipeline."""
        return self.introspector.explain(pipeline_id)

    def pipeline_story(self, pipeline, pipeline_id: str) -> str:
        """Return a Markdown story for a pipeline's evolution and structure."""
        story = PipelineStory(pipeline, self.introspector)
        return story.to_markdown(pipeline_id)
    
    def _get_default_metric(self) -> str:
        """Get the default metric based on the task type."""
        metrics = {
            'classification': 'accuracy',
            'regression': 'neg_mean_squared_error'
        }
        return metrics.get(self.task, 'accuracy')
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **fit_params
    ) -> 'AutoML':
        """
        Fit the AutoML model to the training data.
        
        Args:
            X: Training features.
            y: Training target values.
            X_val: Optional validation features.
            y_val: Optional validation target values.
            **fit_params: Additional fitting parameters.
            
        Returns:
            self: Returns the instance itself.
        """
        self.logger.info("Starting AutoML process")
        
        # Generate initial population of pipelines
        pipelines = self.pipeline_generator.generate_initial_population(
            n_pipelines=self.config.get('n_initial_pipelines', 10)
        )
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population")
        for pipeline in pipelines:
            score = self.evaluator.evaluate(pipeline, X, y, X_val, y_val)
            self.history_.append({
                'pipeline': pipeline,
                'score': score,
                'generation': 0
            })
        
        # Evolutionary optimization
        self.logger.info("Starting evolutionary optimization")
        for gen in range(1, self.config.get('n_generations', 10) + 1):
            # Select best pipelines
            selected = self._select_best_pipelines(
                n_best=self.config.get('n_elite', 3)
            )
            
            # Generate new pipelines through crossover and mutation
            offspring = []
            for _ in range(self.config.get('n_offspring', 5)):
                parent1, parent2 = np.random.choice(selected, 2, replace=False)
                child = self.pipeline_generator.crossover(parent1, parent2)
                child = self.pipeline_generator.mutate(child)
                offspring.append(child)
            
            # Evaluate offspring
            for pipeline in offspring:
                score = self.evaluator.evaluate(pipeline, X, y, X_val, y_val)
                self.history_.append({
                    'pipeline': pipeline,
                    'score': score,
                    'generation': gen
                })
            
            # Update best pipeline
            current_best = max(self.history_, key=lambda x: x['score'])
            if self.best_score_ is None or current_best['score'] > self.best_score_:
                self.best_pipeline_ = current_best['pipeline']
                self.best_score_ = current_best['score']
                self.logger.info(
                    f"Generation {gen}: New best score: {self.best_score_:.4f}"
                )
        
        return self
    
    def _select_best_pipelines(self, n_best: int = 3) -> list:
        """Select the best pipelines from history."""
        sorted_history = sorted(
            self.history_,
            key=lambda x: x['score'],
            reverse=True
        )
        return [item['pipeline'] for item in sorted_history[:n_best]]
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict using the best pipeline found during fit.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted values.
        """
        if self.best_pipeline_ is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' first.")
        return self.best_pipeline_.predict(X)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the AutoML model to disk.
        
        Args:
            path: Path to save the model to.
        """
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info(f"AutoML model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AutoML':
        """
        Load a saved AutoML model from disk.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Loaded AutoML instance.
        """
        import joblib
        path = Path(path)
        automl = joblib.load(path)
        if not isinstance(automl, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return automl
