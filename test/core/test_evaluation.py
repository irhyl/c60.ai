"""
test_evaluation.py — Tests for execution validator, metrics, cache, and fitness evaluator.
"""

import time

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

from c60.core.pipeline import Pipeline, PipelineStep
from c60.core.types import (
    CleanTabular, ScaledData, ClassLabels, RegressionValues,
)
from c60.evaluation.cache import EvaluationCache, FitnessResult
from c60.evaluation.fitness import FitnessEvaluator
from c60.evaluation.metrics import (
    get_metric, available_metrics, accuracy, r2, neg_rmse,
)
from c60.execution.validator import PipelineValidator, ValidationResult


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture
def diabetes():
    return load_diabetes(return_X_y=True)


@pytest.fixture
def clf_pipeline():
    """Minimal valid classification pipeline: Scaler → LogisticRegression."""
    p = Pipeline(name="ClfPipeline")
    scaler = PipelineStep(
        name="Scaler", step_type="scaler",
        operation=StandardScaler(),
        input_type=CleanTabular, output_type=ScaledData,
    )
    clf = PipelineStep(
        name="LR", step_type="classifier",
        operation=LogisticRegression(max_iter=200),
        input_type=ScaledData, output_type=ClassLabels,
    )
    p.add_step(scaler)
    p.add_step(clf)
    p.add_edge(scaler.id, clf.id)
    return p


@pytest.fixture
def reg_pipeline():
    """Minimal valid regression pipeline: Scaler → LinearRegression."""
    p = Pipeline(name="RegPipeline")
    scaler = PipelineStep(
        name="Scaler", step_type="scaler",
        operation=StandardScaler(),
        input_type=CleanTabular, output_type=ScaledData,
    )
    reg = PipelineStep(
        name="LR", step_type="regressor",
        operation=LinearRegression(),
        input_type=ScaledData, output_type=RegressionValues,
    )
    p.add_step(scaler)
    p.add_step(reg)
    p.add_edge(scaler.id, reg.id)
    return p


# ===========================================================================
# Section 1 — Metrics
# ===========================================================================

class TestMetrics:

    def test_accuracy_perfect(self):
        y = np.array([0, 1, 2, 0, 1])
        assert accuracy(y, y) == 1.0

    def test_accuracy_all_wrong(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])
        assert accuracy(y_true, y_pred) == 0.0

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(r2(y, y) - 1.0) < 1e-9

    def test_neg_rmse_is_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert neg_rmse(y_true, y_pred) < 0

    def test_neg_rmse_higher_is_better(self):
        y_true = np.ones(10)
        good_pred = y_true + 0.1
        bad_pred = y_true + 1.0
        assert neg_rmse(y_true, good_pred) > neg_rmse(y_true, bad_pred)

    def test_get_metric_known(self):
        fn = get_metric("accuracy")
        assert callable(fn)

    def test_get_metric_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("unicorn_metric")

    def test_available_metrics_contains_expected(self):
        metrics = available_metrics()
        for expected in ["accuracy", "f1_macro", "r2", "neg_rmse"]:
            assert expected in metrics


# ===========================================================================
# Section 2 — EvaluationCache
# ===========================================================================

class TestEvaluationCache:

    def _make_result(self, score: float = 0.9) -> FitnessResult:
        return FitnessResult(
            raw_score=score,
            adjusted_score=score - 0.01,
            cv_scores=[score] * 5,
            wall_time=1.0,
            n_nodes=3,
        )

    def test_miss_on_empty_cache(self, clf_pipeline):
        cache = EvaluationCache()
        assert cache.get(clf_pipeline) is None

    def test_hit_after_put(self, clf_pipeline):
        cache = EvaluationCache()
        result = self._make_result(0.95)
        cache.put(clf_pipeline, result)
        retrieved = cache.get(clf_pipeline)
        assert retrieved is not None
        assert retrieved.raw_score == 0.95

    def test_contains(self, clf_pipeline):
        cache = EvaluationCache()
        assert not cache.contains(clf_pipeline)
        cache.put(clf_pipeline, self._make_result())
        assert cache.contains(clf_pipeline)

    def test_hit_rate_tracking(self, clf_pipeline):
        cache = EvaluationCache()
        cache.put(clf_pipeline, self._make_result())
        cache.get(clf_pipeline)   # hit
        cache.get(clf_pipeline)   # hit
        p2 = clf_pipeline.clone()
        # Clone has same structure hash → still a hit
        assert cache.get(p2) is not None

    def test_miss_rate_on_new_pipeline(self, clf_pipeline, reg_pipeline):
        cache = EvaluationCache()
        cache.put(clf_pipeline, self._make_result(0.9))
        cache.get(reg_pipeline)  # miss — different structure
        assert cache.misses == 1

    def test_eviction_at_max_size(self):
        cache = EvaluationCache(max_size=2)
        # Create 3 distinct pipelines with different operations
        pipelines = []
        from sklearn.preprocessing import MinMaxScaler, RobustScaler
        for op in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
            p = Pipeline()
            s = PipelineStep(name=op.__class__.__name__, step_type="scaler",
                             operation=op, output_type=ScaledData)
            p.add_step(s)
            pipelines.append(p)

        cache.put(pipelines[0], self._make_result(0.8))
        cache.put(pipelines[1], self._make_result(0.85))
        assert cache.size == 2

        cache.put(pipelines[2], self._make_result(0.9))  # evicts pipelines[0]
        assert cache.size == 2

    def test_clear_resets_everything(self, clf_pipeline):
        cache = EvaluationCache()
        cache.put(clf_pipeline, self._make_result())
        cache.get(clf_pipeline)
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_fitness_result_failed_property(self):
        ok = FitnessResult(0.9, 0.89, [0.9], 1.0, 3)
        assert not ok.failed
        fail = FitnessResult(float("-inf"), float("-inf"), [], 0.5, 3, error="oops")
        assert fail.failed

    def test_stats_string(self, clf_pipeline):
        cache = EvaluationCache()
        cache.put(clf_pipeline, self._make_result())
        s = cache.stats()
        assert "hit_rate" in s


# ===========================================================================
# Section 3 — PipelineValidator
# ===========================================================================

class TestPipelineValidator:

    @pytest.fixture
    def validator(self):
        return PipelineValidator()

    def test_valid_pipeline_passes(self, clf_pipeline, validator):
        result = validator.validate(clf_pipeline)
        assert result.is_valid, str(result)

    def test_empty_pipeline_fails(self, validator):
        result = validator.validate(Pipeline())
        assert not result.is_valid
        assert any("no steps" in e.lower() for e in result.errors)

    def test_disconnected_pipeline_fails(self, validator):
        """Two isolated nodes with no edge between them."""
        p = Pipeline()
        a = PipelineStep(name="A", output_type=ScaledData)
        b = PipelineStep(name="B", input_type=ScaledData, output_type=ClassLabels)
        p.add_step(a)
        p.add_step(b)
        # No edge — both are sources AND sinks; a is not reachable from b
        result = validator.validate(p)
        assert not result.is_valid
        assert any("disconnected" in e.lower() or "not reachable" in e.lower()
                   for e in result.errors)

    def test_single_node_no_predict_warns(self, validator):
        """A single transformer node with no predict — valid structure, warns about predict."""
        p = Pipeline()
        s = PipelineStep(
            name="Scaler", step_type="scaler",
            operation=StandardScaler(),
            input_type=CleanTabular, output_type=ScaledData,
        )
        p.add_step(s)
        result = validator.validate(p)
        # Valid structure (1 node, is its own source and sink)
        # But should warn about no predict() at sink
        assert any("predict" in w.lower() for w in result.warnings)

    def test_validation_result_str(self, clf_pipeline, validator):
        result = validator.validate(clf_pipeline)
        s = str(result)
        assert "ValidationResult" in s

    def test_large_pipeline_warns(self, validator):
        p = Pipeline()
        prev = None
        # Build a chain of 13 steps (above threshold of 12)
        for i in range(13):
            step = PipelineStep(
                name=f"Step{i}",
                input_type=ScaledData,
                output_type=ScaledData,
            )
            p.add_step(step)
            if prev is not None:
                p.add_edge(prev.id, step.id)
            prev = step
        result = validator.validate(p)
        assert any("slow" in w.lower() or "parsimony" in w.lower()
                   for w in result.warnings)


# ===========================================================================
# Section 4 — FitnessEvaluator
# ===========================================================================

class TestFitnessEvaluator:

    def test_classification_returns_valid_score(self, clf_pipeline, iris):
        X, y = iris
        evaluator = FitnessEvaluator(metric="accuracy", cv=3, task="classification")
        result = evaluator.evaluate(clf_pipeline, X, y)
        assert not result.failed, f"Unexpected failure: {result.error}"
        assert 0.0 <= result.raw_score <= 1.0
        assert len(result.cv_scores) == 3

    def test_regression_returns_valid_score(self, reg_pipeline, diabetes):
        X, y = diabetes
        evaluator = FitnessEvaluator(metric="r2", cv=3, task="regression")
        result = evaluator.evaluate(reg_pipeline, X, y)
        assert not result.failed, f"Unexpected failure: {result.error}"
        assert result.raw_score <= 1.0   # r2 can be negative for bad models

    def test_adjusted_score_less_than_raw_with_penalty(self, clf_pipeline, iris):
        X, y = iris
        evaluator = FitnessEvaluator(
            metric="accuracy", cv=3, task="classification",
            complexity_penalty=0.05,
        )
        result = evaluator.evaluate(clf_pipeline, X, y)
        assert result.adjusted_score < result.raw_score

    def test_zero_penalty_adjusted_equals_raw(self, clf_pipeline, iris):
        X, y = iris
        evaluator = FitnessEvaluator(
            metric="accuracy", cv=3, task="classification",
            complexity_penalty=0.0,
        )
        result = evaluator.evaluate(clf_pipeline, X, y)
        assert abs(result.adjusted_score - result.raw_score) < 1e-9

    def test_broken_pipeline_returns_neg_inf(self, iris):
        """A pipeline with no valid sink estimator should fail gracefully."""
        X, y = iris
        # Pipeline with only a scaler — sink has no predict()
        # The validator catches this as a warning but the CV will fail because
        # scaler.predict() doesn't exist in the expected way.
        # We use an empty pipeline to force a validation failure.
        empty = Pipeline(name="Empty")
        evaluator = FitnessEvaluator(metric="accuracy", cv=3, task="classification")
        result = evaluator.evaluate(empty, X, y)
        assert result.failed
        assert result.adjusted_score == float("-inf")

    def test_wall_time_is_recorded(self, clf_pipeline, iris):
        X, y = iris
        evaluator = FitnessEvaluator(metric="accuracy", cv=3, task="classification")
        result = evaluator.evaluate(clf_pipeline, X, y)
        assert result.wall_time > 0

    def test_cache_prevents_re_evaluation(self, clf_pipeline, iris):
        X, y = iris
        cache = EvaluationCache()
        evaluator = FitnessEvaluator(
            metric="accuracy", cv=3, task="classification", cache=cache
        )
        result1 = evaluator.evaluate(clf_pipeline, X, y)

        # Second call on structurally identical pipeline (clone) — should hit cache
        cloned = clf_pipeline.clone()
        result2 = evaluator.evaluate(cloned, X, y)

        # Score must be identical (served from cache, no new CV run)
        assert result2.raw_score == result1.raw_score
        assert result2.cv_scores == result1.cv_scores
        # Cache must have recorded at least one hit
        assert cache.hits >= 1

    def test_iris_accuracy_above_threshold(self, clf_pipeline, iris):
        """Sanity check: Scaler + LR on Iris should consistently beat 90%."""
        X, y = iris
        evaluator = FitnessEvaluator(metric="accuracy", cv=5, task="classification")
        result = evaluator.evaluate(clf_pipeline, X, y)
        assert result.raw_score > 0.90, (
            f"Expected >90% accuracy on Iris, got {result.raw_score:.2%}"
        )

    def test_n_nodes_recorded_correctly(self, clf_pipeline, iris):
        X, y = iris
        evaluator = FitnessEvaluator(metric="accuracy", cv=3, task="classification")
        result = evaluator.evaluate(clf_pipeline, X, y)
        assert result.n_nodes == len(clf_pipeline)

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="task must be"):
            FitnessEvaluator(task="space_travel")

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            FitnessEvaluator(metric="moon_score", task="classification")
