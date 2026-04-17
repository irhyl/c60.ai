"""
introspector.py — PipelineIntrospector: extract interpretability from a fitted pipeline.

For each step in the topological order the introspector tries to pull
whatever interpretability signal is available from the sklearn estimator:

  feature_importances_      — tree-based models (RF, GBT, ExtraTree, DT)
  coef_                     — linear models (LogisticRegression, Ridge, SVM, …)
  explained_variance_ratio_ — PCA
  scores_                   — univariate selectors (SelectKBest, SelectPercentile)

Multi-class coef_ tensors (shape n_classes × n_features) are collapsed to
mean-absolute-value per feature.  All importances are L1-normalised to [0, 1]
so they represent a probability-like share of total importance.

The pipeline must be fitted before inspect() is called.  Calling inspect() on
an unfitted pipeline is not an error — importance fields will simply be None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from c60.core.pipeline import Pipeline, PipelineStep
from c60.evaluation.cache import FitnessResult


# ---------------------------------------------------------------------------
# StepReport — interpretability snapshot for one step
# ---------------------------------------------------------------------------

@dataclass
class StepReport:
    """Interpretability snapshot for one pipeline step."""

    step_id: str
    step_name: str
    step_type: str
    operation_class: str
    params: Dict
    feature_importances: Optional[np.ndarray] = field(default=None, repr=False)
    importance_source: Optional[str] = None   # e.g. "feature_importances_"
    n_input_features: Optional[int] = None
    n_output_features: Optional[int] = None

    def has_importances(self) -> bool:
        return self.feature_importances is not None

    def top_k_indices(self, k: int = 5) -> List[int]:
        """Return indices of the k most important features (descending)."""
        if self.feature_importances is None:
            return []
        k = min(k, len(self.feature_importances))
        return list(np.argsort(self.feature_importances)[::-1][:k])

    def __repr__(self) -> str:
        src = f", src={self.importance_source}" if self.importance_source else ""
        return (
            f"StepReport({self.step_name!r}, type={self.step_type!r}"
            f"{src})"
        )


# ---------------------------------------------------------------------------
# PipelineReport — complete interpretability report
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    """Complete interpretability report for one (optionally fitted) pipeline."""

    pipeline_name: str
    n_steps: int
    steps: List[StepReport] = field(default_factory=list)
    fitness: Optional[FitnessResult] = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def final_step(self) -> Optional[StepReport]:
        """Return the last step in topological order."""
        return self.steps[-1] if self.steps else None

    def predictive_step(self) -> Optional[StepReport]:
        """
        Return the step most useful for feature importance reporting.

        Preference order:
          1. A classifier/regressor with importances
          2. Any step with importances (e.g. PCA, SelectKBest)
        """
        for step in reversed(self.steps):
            if step.has_importances() and step.step_type in ("classifier", "regressor"):
                return step
        for step in reversed(self.steps):
            if step.has_importances():
                return step
        return None

    def top_features(
        self,
        feature_names: Optional[List[str]] = None,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Return up to k (feature_name, importance) pairs from the most
        informative step.

        If feature_names is None, labels are "feature_0", "feature_1", …
        """
        step = self.predictive_step()
        if step is None or step.feature_importances is None:
            return []
        imp = step.feature_importances
        indices = step.top_k_indices(k)
        names = feature_names or [f"feature_{i}" for i in range(len(imp))]
        return [
            (names[i] if i < len(names) else f"feature_{i}", float(imp[i]))
            for i in indices
        ]

    def summary(self) -> str:
        """Return a multi-line human-readable summary."""
        lines = [
            f"PipelineReport: '{self.pipeline_name}' ({self.n_steps} steps)"
        ]
        if self.fitness:
            lines.append(
                f"  Fitness: raw={self.fitness.raw_score:.4f}, "
                f"adj={self.fitness.adjusted_score:.4f}"
            )
        for step in self.steps:
            src = f" [{step.importance_source}]" if step.importance_source else ""
            ni = f", in={step.n_input_features}" if step.n_input_features else ""
            no = f", out={step.n_output_features}" if step.n_output_features else ""
            lines.append(
                f"  [{step.step_type:20s}] {step.step_name}{src}{ni}{no}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        score = (
            f"{self.fitness.adjusted_score:.4f}" if self.fitness else "unevaluated"
        )
        return (
            f"PipelineReport(name={self.pipeline_name!r}, "
            f"steps={self.n_steps}, score={score})"
        )


# ---------------------------------------------------------------------------
# PipelineIntrospector
# ---------------------------------------------------------------------------

class PipelineIntrospector:
    """
    Extract interpretability information from a Pipeline.

    Usage
    -----
    introspector = PipelineIntrospector()
    report = introspector.inspect(fitted_pipeline, fitness=fitness_result)
    print(report.summary())
    top = report.top_features(feature_names=["sepal_len", ...], k=4)
    """

    # Extraction strategies tried in order.  Each entry is
    # (attribute_name, label_used_in_importance_source).
    _IMPORTANCE_ATTRS: List[Tuple[str, str]] = [
        ("feature_importances_", "feature_importances_"),
        ("coef_",                "coef_"),
        ("explained_variance_ratio_", "explained_variance_ratio_"),
        ("scores_",              "scores_"),
    ]

    def inspect(
        self,
        pipeline: Pipeline,
        fitness: Optional[FitnessResult] = None,
    ) -> PipelineReport:
        """
        Inspect a pipeline and return a PipelineReport.

        The pipeline should be fitted for importances to be populated.
        Calling on an unfitted pipeline is safe — importance fields are None.
        """
        report = PipelineReport(
            pipeline_name=pipeline.name or "unnamed",
            n_steps=len(pipeline),
            fitness=fitness,
        )
        for step in pipeline.topological_order():
            report.steps.append(self._inspect_step(step))
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inspect_step(self, step: PipelineStep) -> StepReport:
        op = step.operation
        importances: Optional[np.ndarray] = None
        source: Optional[str] = None
        n_in: Optional[int] = None
        n_out: Optional[int] = None

        if op is not None:
            importances, source = self._extract_importances(op)
            n_in = getattr(op, "n_features_in_", None)
            n_out = self._infer_n_output(op)

        return StepReport(
            step_id=step.id,
            step_name=step.name,
            step_type=step.step_type,
            operation_class=type(op).__name__ if op else "None",
            params=dict(step.params),
            feature_importances=importances,
            importance_source=source,
            n_input_features=n_in,
            n_output_features=n_out,
        )

    def _extract_importances(
        self, op: object
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Try each known attribute in turn; return normalised array + label."""
        for attr, label in self._IMPORTANCE_ATTRS:
            raw = getattr(op, attr, None)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=float)
            # Collapse 2-D coef_ (multi-class): mean |coef| per feature
            if arr.ndim == 2:
                arr = np.mean(np.abs(arr), axis=0)
            arr = np.abs(arr)
            total = arr.sum()
            if total > 0:
                arr = arr / total
            return arr, label
        return None, None

    def _infer_n_output(self, op: object) -> Optional[int]:
        """Heuristic: how many features does this step emit?"""
        if hasattr(op, "n_components_"):
            return int(op.n_components_)
        if hasattr(op, "get_support"):
            try:
                return int(op.get_support().sum())
            except Exception:
                pass
        return None
