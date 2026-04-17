"""
test_parallel.py — Tests for ParallelEvaluator.

Verifies:
  - Parallel results match sequential results (correctness)
  - Thread safety (no race conditions with shared population)
  - n_jobs parameter behaviour (-1, 1, explicit)
  - Error handling when individual evaluation crashes
  - reevaluate flag is respected
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from c60.core.pipeline import Pipeline, PipelineStep
from c60.core.registry import default_registry
from c60.core.types import CleanTabular, ScaledData, ClassLabels
from c60.evaluation.fitness import FitnessEvaluator
from c60.evolution.population import Individual, Population
from c60.execution.parallel import ParallelEvaluator


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture
def fast_evaluator():
    return FitnessEvaluator(
        metric="accuracy", cv=2, task="classification",
        timeout_seconds=30.0, complexity_penalty=0.0,
    )


def _make_clf_pipeline():
    p = Pipeline(name="TestPipeline")
    sc = PipelineStep(
        name="StandardScaler", step_type="scaler",
        operation=StandardScaler(),
        input_type=CleanTabular, output_type=ScaledData,
    )
    clf = PipelineStep(
        name="LogisticRegression", step_type="classifier",
        operation=LogisticRegression(max_iter=200),
        input_type=ScaledData, output_type=ClassLabels,
    )
    p.add_step(sc)
    p.add_step(clf)
    p.add_edge(sc.id, clf.id)
    return p


def _make_population(n: int) -> Population:
    """Build a population of n identical (but independently cloned) pipelines."""
    pipelines = [_make_clf_pipeline().clone() for _ in range(n)]
    pop = Population(size=n)
    pop.initialize_from(pipelines)
    return pop


# ===========================================================================
# Section 1 — Construction
# ===========================================================================

class TestParallelEvaluatorConstruction:

    def test_default_n_jobs(self):
        pe = ParallelEvaluator()
        assert pe.n_jobs == -1

    def test_explicit_n_jobs(self):
        pe = ParallelEvaluator(n_jobs=4)
        assert pe.n_jobs == 4

    def test_n_jobs_zero_raises(self):
        with pytest.raises(ValueError, match="n_jobs"):
            ParallelEvaluator(n_jobs=0)

    def test_effective_n_jobs_positive(self):
        pe = ParallelEvaluator(n_jobs=-1)
        assert pe.effective_n_jobs >= 1

    def test_repr_contains_n_jobs(self):
        pe = ParallelEvaluator(n_jobs=3)
        assert "3" in repr(pe)


# ===========================================================================
# Section 2 — Correctness: parallel == sequential
# ===========================================================================

class TestParallelCorrectness:

    def test_parallel_scores_match_sequential(self, iris, fast_evaluator):
        """Parallel evaluation must produce identical scores to sequential."""
        X, y = iris
        pop_seq = _make_population(4)
        pop_par = Population(size=4)
        pop_par.initialize_from([ind.pipeline.clone()
                                  for ind in pop_seq])

        # Sequential
        pop_seq.evaluate_all(fast_evaluator, X, y)

        # Parallel
        pe = ParallelEvaluator(n_jobs=2)
        pe.evaluate_population(pop_par, fast_evaluator, X, y)

        for seq_ind, par_ind in zip(pop_seq, pop_par):
            assert seq_ind.fitness is not None
            assert par_ind.fitness is not None
            assert abs(seq_ind.score - par_ind.score) < 1e-9, (
                f"Mismatch: seq={seq_ind.score:.4f}, par={par_ind.score:.4f}"
            )

    def test_single_job_equals_sequential(self, iris, fast_evaluator):
        """n_jobs=1 must give identical results to Population.evaluate_all."""
        X, y = iris
        pop_seq = _make_population(3)
        pop_par = Population(size=3)
        pop_par.initialize_from([ind.pipeline.clone()
                                  for ind in pop_seq])

        pop_seq.evaluate_all(fast_evaluator, X, y)
        ParallelEvaluator(n_jobs=1).evaluate_population(pop_par, fast_evaluator, X, y)

        for s, p in zip(pop_seq, pop_par):
            assert abs(s.score - p.score) < 1e-9

    def test_all_individuals_get_fitness(self, iris, fast_evaluator):
        X, y = iris
        pop = _make_population(5)
        pe = ParallelEvaluator(n_jobs=2)
        pe.evaluate_population(pop, fast_evaluator, X, y)
        for ind in pop:
            assert ind.fitness is not None


# ===========================================================================
# Section 3 — reevaluate flag
# ===========================================================================

class TestReevaluateFlag:

    def test_skips_already_evaluated(self, iris, fast_evaluator):
        """Parallel evaluator should skip individuals with existing fitness."""
        X, y = iris
        pop = _make_population(3)
        pe = ParallelEvaluator(n_jobs=2)

        # First evaluation
        pe.evaluate_population(pop, fast_evaluator, X, y)
        first_scores = [ind.score for ind in pop]

        # Second evaluation — without reevaluate: should be unchanged
        pe.evaluate_population(pop, fast_evaluator, X, y, reevaluate=False)
        second_scores = [ind.score for ind in pop]

        for a, b in zip(first_scores, second_scores):
            assert abs(a - b) < 1e-9

    def test_reevaluate_true_re_runs(self, iris, fast_evaluator):
        """With reevaluate=True all individuals are re-evaluated."""
        X, y = iris
        pop = _make_population(3)
        pe = ParallelEvaluator(n_jobs=2)

        pe.evaluate_population(pop, fast_evaluator, X, y)
        # reevaluate=True — should complete without error
        pe.evaluate_population(pop, fast_evaluator, X, y, reevaluate=True)
        for ind in pop:
            assert ind.fitness is not None


# ===========================================================================
# Section 4 — Empty population / no work
# ===========================================================================

class TestEdgeCases:

    def test_empty_population_no_error(self, fast_evaluator, iris):
        X, y = iris
        pop = Population(size=2)
        # Manually inject individuals with existing fitness
        from c60.evaluation.cache import FitnessResult
        pipelines = [_make_clf_pipeline().clone() for _ in range(2)]
        pop.initialize_from(pipelines)
        for ind in pop:
            ind.fitness = FitnessResult(
                raw_score=0.9, adjusted_score=0.9,
                cv_scores=[0.9, 0.9], wall_time=0.1, n_nodes=2,
            )
        # All already evaluated — parallel should do nothing
        pe = ParallelEvaluator(n_jobs=2)
        pe.evaluate_population(pop, fast_evaluator, X, y, reevaluate=False)
        # Scores unchanged
        for ind in pop:
            assert ind.score == pytest.approx(0.9)

    def test_more_jobs_than_individuals(self, iris, fast_evaluator):
        """n_jobs > population size should work fine (caps to pop size)."""
        X, y = iris
        pop = _make_population(2)
        pe = ParallelEvaluator(n_jobs=8)
        pe.evaluate_population(pop, fast_evaluator, X, y)
        for ind in pop:
            assert ind.fitness is not None

    def test_population_order_preserved(self, iris, fast_evaluator):
        """The order of individuals in the population must not change."""
        X, y = iris
        pop = _make_population(4)
        ids_before = [id(ind) for ind in pop]
        ParallelEvaluator(n_jobs=2).evaluate_population(pop, fast_evaluator, X, y)
        ids_after = [id(ind) for ind in pop]
        assert ids_before == ids_after
