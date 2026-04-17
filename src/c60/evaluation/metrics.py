"""
metrics.py — Metric functions for pipeline fitness evaluation.

Each metric function has the signature:
    metric(y_true, y_pred) -> float

Higher is always better (maximization convention throughout C60.ai).
For metrics like RMSE where lower is better, we negate the value.

Supported metrics
-----------------
Classification : accuracy, f1_macro, f1_weighted, roc_auc, balanced_accuracy
Regression     : r2, neg_rmse, neg_mae

Usage
-----
    from c60.evaluation.metrics import get_metric
    score_fn = get_metric("accuracy")
    score = score_fn(y_true, y_pred)
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# Type alias: a metric is any callable (y_true, y_pred) -> float
MetricFn = Callable[[np.ndarray, np.ndarray], float]


# ---------------------------------------------------------------------------
# Classification metrics (all return higher-is-better floats)
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ROC-AUC using predict_proba output.
    y_pred should be probability estimates (n_samples × n_classes) for
    multi-class, or (n_samples,) probabilities for binary.
    Falls back to accuracy if shape is wrong.
    """
    try:
        if y_pred.ndim == 2:
            return float(
                roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
            )
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        # Graceful fallback: roc_auc_score can fail on degenerate predictions
        return float(accuracy_score(y_true, y_pred.argmax(axis=1) if y_pred.ndim == 2 else y_pred))


# ---------------------------------------------------------------------------
# Regression metrics (negated so higher is always better)
# ---------------------------------------------------------------------------

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def neg_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative RMSE — higher (less negative) means better."""
    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


def neg_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative MAE — higher (less negative) means better."""
    return -float(mean_absolute_error(y_true, y_pred))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_METRICS: Dict[str, MetricFn] = {
    # classification
    "accuracy":          accuracy,
    "f1_macro":          f1_macro,
    "f1_weighted":       f1_weighted,
    "balanced_accuracy": balanced_accuracy,
    "roc_auc":           roc_auc,
    # regression
    "r2":                r2,
    "neg_rmse":          neg_rmse,
    "neg_mae":           neg_mae,
}

#: Map metric name → task type (for documentation / validation)
METRIC_TASK: Dict[str, str] = {
    "accuracy":          "classification",
    "f1_macro":          "classification",
    "f1_weighted":       "classification",
    "balanced_accuracy": "classification",
    "roc_auc":           "classification",
    "r2":                "regression",
    "neg_rmse":          "regression",
    "neg_mae":           "regression",
}

#: Default metric per task type
DEFAULT_METRIC: Dict[str, str] = {
    "classification": "accuracy",
    "regression":     "r2",
}


def get_metric(name: str) -> MetricFn:
    """
    Return the metric function for the given name.

    Raises
    ------
    ValueError
        If the metric name is not recognised.
    """
    if name not in _METRICS:
        raise ValueError(
            f"Unknown metric '{name}'. "
            f"Available: {sorted(_METRICS.keys())}"
        )
    return _METRICS[name]


def available_metrics() -> list[str]:
    """Return the sorted list of all registered metric names."""
    return sorted(_METRICS.keys())
