"""
fitness.py — Pipeline fitness evaluation via cross-validation.

FitnessEvaluator is the bridge between a Pipeline and a numeric score.
It wraps sklearn cross-validation, enforces a timeout, applies a parsimony
penalty for pipeline complexity, and returns a FitnessResult.

Design decisions
----------------
- Broken pipelines (fit errors, shape mismatches, timeouts) return
  adjusted_score = float('-inf') rather than raising.  The evolution
  loop must never crash because one candidate failed.
- The evaluator is stateless: each call is independent.  Caching is
  handled externally by EvaluationCache.
- Timeout is implemented via threading.  On Windows, signal-based
  timeouts are unavailable, so we use a Thread + join(timeout).
- The parsimony penalty uses normalised pipeline size:
      adjusted = raw - lambda * (n_nodes / max_pipeline_size)
  so lambda is interpretable as "how much accuracy we sacrifice for
  each additional node beyond max_pipeline_size".
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from c60.core.pipeline import Pipeline
from c60.evaluation.cache import EvaluationCache, FitnessResult
from c60.evaluation.metrics import get_metric, DEFAULT_METRIC
from c60.execution.validator import PipelineValidator


# ---------------------------------------------------------------------------
# FitnessEvaluator
# ---------------------------------------------------------------------------

class FitnessEvaluator:
    """
    Evaluates a pipeline's fitness on a dataset using K-fold cross-validation.

    Parameters
    ----------
    metric : str
        Metric name from c60.evaluation.metrics (e.g. "accuracy", "r2").
    cv : int
        Number of cross-validation folds.
    task : str
        "classification" or "regression".  Determines fold strategy
        (StratifiedKFold vs KFold) and the default metric.
    timeout_seconds : float
        Maximum wall-clock time per evaluation.  Exceeded evaluations
        return float('-inf') fitness.
    complexity_penalty : float
        Lambda in: adjusted = raw - lambda * (n_nodes / max_pipeline_nodes).
        Set to 0.0 to disable parsimony pressure.
    max_pipeline_nodes : int
        Normalisation denominator for the complexity penalty.
    cache : EvaluationCache or None
        If provided, results are looked up and stored here.
    """

    def __init__(
        self,
        metric: Optional[str] = None,
        cv: int = 5,
        task: str = "classification",
        timeout_seconds: float = 120.0,
        complexity_penalty: float = 0.01,
        max_pipeline_nodes: int = 15,
        cache: Optional[EvaluationCache] = None,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
        self.task = task
        self.metric_name: str = metric or DEFAULT_METRIC[task]
        self.metric_fn = get_metric(self.metric_name)
        self.cv = cv
        self.timeout_seconds = timeout_seconds
        self.complexity_penalty = complexity_penalty
        self.max_pipeline_nodes = max_pipeline_nodes
        self.cache = cache
        self._validator = PipelineValidator()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, pipeline: Pipeline, X: Any, y: Any) -> FitnessResult:
        """
        Score the pipeline on (X, y).

        Returns a FitnessResult.  Never raises — broken pipelines get
        adjusted_score = float('-inf') and a populated .error field.

        Checks the cache first if one was provided.
        """
        # Cache lookup
        if self.cache is not None:
            cached = self.cache.get(pipeline)
            if cached is not None:
                return cached

        result = self._run_evaluation(pipeline, X, y)

        # Store in cache
        if self.cache is not None:
            self.cache.put(pipeline, result)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_evaluation(self, pipeline: Pipeline, X: Any, y: Any) -> FitnessResult:
        t_start = time.perf_counter()

        # 1. Structural validation — fast check before any fitting
        val_result = self._validator.validate(pipeline)
        if not val_result.is_valid:
            error_msg = "; ".join(val_result.errors)
            return self._failure_result(
                error=f"Validation failed: {error_msg}",
                n_nodes=len(pipeline),
                wall_time=time.perf_counter() - t_start,
            )

        # 2. Run cross-validation in a thread so we can enforce timeout
        cv_scores: list[float] = []
        exception_holder: list[str] = []

        def _cv_worker() -> None:
            try:
                cv_scores.extend(
                    self._cross_validate(pipeline, X, y)
                )
            except Exception as exc:
                exception_holder.append(
                    f"{type(exc).__name__}: {exc}\n"
                    + traceback.format_exc()
                )

        thread = threading.Thread(target=_cv_worker, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        wall_time = time.perf_counter() - t_start

        if thread.is_alive():
            # Timeout: thread is still running
            return self._failure_result(
                error=f"Timeout after {self.timeout_seconds:.0f}s",
                n_nodes=len(pipeline),
                wall_time=wall_time,
            )

        if exception_holder:
            return self._failure_result(
                error=exception_holder[0][:300],  # truncate long tracebacks
                n_nodes=len(pipeline),
                wall_time=wall_time,
            )

        if not cv_scores:
            return self._failure_result(
                error="Cross-validation returned no scores.",
                n_nodes=len(pipeline),
                wall_time=wall_time,
            )

        # 3. Compute adjusted score with parsimony penalty
        raw_score = float(np.mean(cv_scores))
        penalty = self.complexity_penalty * (
            len(pipeline) / max(self.max_pipeline_nodes, 1)
        )
        adjusted_score = raw_score - penalty

        return FitnessResult(
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            cv_scores=list(cv_scores),
            wall_time=wall_time,
            n_nodes=len(pipeline),
        )

    def _cross_validate(
        self, pipeline: Pipeline, X: Any, y: Any
    ) -> list[float]:
        """
        Run K-fold CV manually so we can use Pipeline.fit/predict directly
        (sklearn cross_val_score expects a sklearn estimator interface, which
        Pipeline does not fully implement — e.g. get_params / set_params).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.task == "classification":
            splitter = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=42
            )
        else:
            splitter = KFold(
                n_splits=self.cv, shuffle=True, random_state=42
            )

        fold_scores = []
        for train_idx, val_idx in splitter.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Clone the pipeline for each fold so folds are independent
            fold_pipeline = pipeline.clone()
            fold_pipeline.fit(X_train, y_train)
            y_pred = fold_pipeline.predict(X_val)

            score = self.metric_fn(y_val, y_pred)
            fold_scores.append(score)

        return fold_scores

    @staticmethod
    def _failure_result(
        error: str,
        n_nodes: int,
        wall_time: float,
    ) -> FitnessResult:
        return FitnessResult(
            raw_score=float("-inf"),
            adjusted_score=float("-inf"),
            cv_scores=[],
            wall_time=wall_time,
            n_nodes=n_nodes,
            error=error,
        )
