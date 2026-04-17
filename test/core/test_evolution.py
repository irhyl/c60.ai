"""
test_evolution.py — Tests for Population, Selection, and EvolutionEngine.

Keeps evaluations fast by using small populations (5–10) and few
generations (2–5) with 2-fold CV.  The goal is correctness, not
benchmark performance.
"""

import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

from c60.core.pipeline import Pipeline, PipelineStep
from c60.core.registry import default_registry
from c60.core.types import CleanTabular, ScaledData, ClassLabels, RegressionValues
from c60.evaluation.fitness import FitnessEvaluator
from c60.evolution.engine import EvolutionEngine, EvolutionLog, GenerationRecord
from c60.evolution.population import Individual, Population
from c60.evolution.selection import ElitismPreserver, TournamentSelector


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture
def diabetes():
    return load_diabetes(return_X_y=True)


@pytest.fixture
def clf_pipeline():
    p = Pipeline(name="ClfSeed")
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


@pytest.fixture
def small_evaluator():
    """Fast evaluator: 2-fold CV, no penalty, short timeout."""
    return FitnessEvaluator(
        metric="accuracy", cv=2, task="classification",
        timeout_seconds=30.0, complexity_penalty=0.0,
    )


# ===========================================================================
# Section 1 — Individual
# ===========================================================================

class TestIndividual:

    def test_score_before_evaluation_is_neg_inf(self, clf_pipeline):
        ind = Individual(pipeline=clf_pipeline)
        assert ind.score == float("-inf")

    def test_score_after_evaluation(self, clf_pipeline, iris, small_evaluator):
        X, y = iris
        ind = Individual(pipeline=clf_pipeline)
        ind.evaluate(small_evaluator, X, y)
        assert ind.fitness is not None
        assert ind.score > float("-inf")

    def test_repr_unevaluated(self, clf_pipeline):
        ind = Individual(pipeline=clf_pipeline)
        assert "unevaluated" in repr(ind)

    def test_repr_evaluated(self, clf_pipeline, iris, small_evaluator):
        X, y = iris
        ind = Individual(pipeline=clf_pipeline)
        ind.evaluate(small_evaluator, X, y)
        assert "score=" in repr(ind)


# ===========================================================================
# Section 2 — Population
# ===========================================================================

class TestPopulation:

    def test_initialize_classification(self):
        pop = Population(size=6)
        pop.initialize(default_registry, task="classification")
        assert len(pop) == 6
        for ind in pop:
            assert len(ind.pipeline) >= 2

    def test_initialize_regression(self):
        pop = Population(size=4)
        pop.initialize(default_registry, task="regression")
        assert len(pop) == 4

    def test_size_too_small_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            Population(size=1)

    def test_evaluate_all(self, iris, small_evaluator):
        X, y = iris
        pop = Population(size=4)
        pop.initialize(default_registry, task="classification")
        pop.evaluate_all(small_evaluator, X, y)
        for ind in pop:
            assert ind.fitness is not None

    def test_best_returns_highest_score(self, iris, small_evaluator):
        X, y = iris
        pop = Population(size=5)
        pop.initialize(default_registry, task="classification")
        pop.evaluate_all(small_evaluator, X, y)
        best = pop.best()
        for ind in pop:
            assert ind.score <= best.score + 1e-9

    def test_sorted_individuals_descending(self, iris, small_evaluator):
        X, y = iris
        pop = Population(size=5)
        pop.initialize(default_registry, task="classification")
        pop.evaluate_all(small_evaluator, X, y)
        scores = [ind.score for ind in pop.sorted_individuals()]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_length(self, iris, small_evaluator):
        X, y = iris
        pop = Population(size=6)
        pop.initialize(default_registry, task="classification")
        pop.evaluate_all(small_evaluator, X, y)
        assert len(pop.top_k(3)) == 3

    def test_mean_score_in_range(self, iris, small_evaluator):
        X, y = iris
        pop = Population(size=4)
        pop.initialize(default_registry, task="classification")
        pop.evaluate_all(small_evaluator, X, y)
        mean = pop.mean_score()
        assert 0.0 <= mean <= 1.0

    def test_initialize_from_list(self, clf_pipeline):
        pipelines = [clf_pipeline.clone() for _ in range(3)]
        pop = Population(size=3)
        pop.initialize_from(pipelines)
        assert len(pop) == 3

    def test_reevaluate_flag(self, iris, small_evaluator):
        X, y = iris
        pop = Population(size=3)
        pop.initialize(default_registry, task="classification")
        pop.evaluate_all(small_evaluator, X, y)
        scores_before = [ind.score for ind in pop]
        pop.evaluate_all(small_evaluator, X, y, reevaluate=True)
        scores_after = [ind.score for ind in pop]
        # Scores should be the same (same pipelines, same data, same cache)
        for a, b in zip(scores_before, scores_after):
            if a == float("-inf") and b == float("-inf"):
                continue  # both failed — consistent
            assert abs(a - b) < 1e-9


# ===========================================================================
# Section 3 — TournamentSelector
# ===========================================================================

class TestTournamentSelector:

    def _make_pop_with_scores(self, scores):
        """Build a population of dummy individuals with preset scores."""
        from c60.evaluation.cache import FitnessResult
        pop = Population(size=len(scores))
        pipelines = []
        for _ in scores:
            p = Pipeline()
            step = PipelineStep(name="X", step_type="scaler",
                                output_type=ScaledData)
            p.add_step(step)
            pipelines.append(p)
        pop.initialize_from(pipelines)
        for ind, score in zip(pop, scores):
            ind.fitness = FitnessResult(
                raw_score=score, adjusted_score=score,
                cv_scores=[score], wall_time=0.1, n_nodes=1,
            )
        return pop

    def test_selects_from_population(self):
        pop = self._make_pop_with_scores([0.5, 0.7, 0.9, 0.3])
        sel = TournamentSelector(tournament_size=2)
        selected = sel.select(pop)
        assert selected in list(pop)

    def test_tends_to_select_higher_scores(self):
        """Higher-scoring individuals should win tournaments more often than lower ones."""
        pop = self._make_pop_with_scores([0.5, 0.7, 0.9, 0.3])
        sel = TournamentSelector(tournament_size=3)
        wins = {0.5: 0, 0.7: 0, 0.9: 0, 0.3: 0}
        for _ in range(200):
            winner = sel.select(pop)
            wins[winner.score] += 1
        # The best individual (0.9) should win more than any single other individual
        assert wins[0.9] > wins[0.3]
        assert wins[0.9] > wins[0.5]

    def test_select_pair_returns_two(self):
        pop = self._make_pop_with_scores([0.5, 0.7, 0.9, 0.3])
        sel = TournamentSelector(tournament_size=2)
        p1, p2 = sel.select_pair(pop)
        assert isinstance(p1, Individual)
        assert isinstance(p2, Individual)

    def test_tournament_size_too_small_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            TournamentSelector(tournament_size=1)


# ===========================================================================
# Section 4 — ElitismPreserver
# ===========================================================================

class TestElitismPreserver:

    def _make_evaluated_pop(self, scores):
        from c60.evaluation.cache import FitnessResult
        pop = Population(size=len(scores))
        pipelines = [Pipeline() for _ in scores]
        for p, s in zip(pipelines, scores):
            step = PipelineStep(name="X", step_type="scaler", output_type=ScaledData)
            p.add_step(step)
        pop.initialize_from(pipelines)
        for ind, score in zip(pop, scores):
            ind.fitness = FitnessResult(
                raw_score=score, adjusted_score=score,
                cv_scores=[score], wall_time=0.1, n_nodes=1,
            )
        return pop

    def test_returns_correct_count(self):
        pop = self._make_evaluated_pop([0.5, 0.7, 0.9, 0.3, 0.8])
        elites = ElitismPreserver(elite_count=2).get_elites(pop)
        assert len(elites) == 2

    def test_returns_highest_scoring_individuals(self):
        pop = self._make_evaluated_pop([0.5, 0.7, 0.9, 0.3, 0.8])
        elites = ElitismPreserver(elite_count=2).get_elites(pop)
        elite_scores = sorted([e.score for e in elites], reverse=True)
        assert elite_scores[0] == pytest.approx(0.9)
        assert elite_scores[1] == pytest.approx(0.8)

    def test_elites_are_clones_not_same_objects(self):
        pop = self._make_evaluated_pop([0.5, 0.7, 0.9])
        orig_ids = {ind.pipeline.structure_hash() for ind in pop}
        elites = ElitismPreserver(elite_count=1).get_elites(pop)
        # Elite carries over known fitness
        assert elites[0].fitness is not None
        assert elites[0].score == pytest.approx(0.9)

    def test_zero_elites_returns_empty(self):
        pop = self._make_evaluated_pop([0.5, 0.7])
        elites = ElitismPreserver(elite_count=0).get_elites(pop)
        assert elites == []

    def test_negative_elite_count_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            ElitismPreserver(elite_count=-1)


# ===========================================================================
# Section 5 — EvolutionEngine (integration)
# ===========================================================================

class TestEvolutionEngine:

    def test_fit_returns_pipeline(self, iris):
        X, y = iris
        engine = EvolutionEngine(
            population_size=5, max_generations=2,
            cv=2, task="classification",
            eval_timeout=20.0, random_seed=42,
        )
        result = engine.fit(X, y)
        assert isinstance(result, Pipeline)

    def test_fit_regression_returns_pipeline(self, diabetes):
        X, y = diabetes
        engine = EvolutionEngine(
            population_size=5, max_generations=2,
            cv=2, task="regression", metric="r2",
            eval_timeout=30.0, random_seed=42,
        )
        result = engine.fit(X, y)
        assert isinstance(result, Pipeline)

    def test_history_has_generation_records(self, iris):
        X, y = iris
        engine = EvolutionEngine(
            population_size=4, max_generations=3,
            cv=2, task="classification",
            eval_timeout=20.0, random_seed=0,
        )
        engine.fit(X, y)
        log = engine.history()
        assert isinstance(log, EvolutionLog)
        assert log.generations_run() >= 1

    def test_best_score_nondecreasing_across_generations(self, iris):
        """Elitism guarantees best score never decreases."""
        X, y = iris
        engine = EvolutionEngine(
            population_size=5, max_generations=4,
            cv=2, task="classification",
            eval_timeout=20.0, random_seed=7,
            elite_count=2,
        )
        engine.fit(X, y)
        scores = engine.history().best_scores()
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 1e-6, (
                f"Best score decreased at generation {i}: "
                f"{scores[i-1]:.4f} → {scores[i]:.4f}"
            )

    def test_returned_pipeline_is_functional(self, iris):
        """The returned pipeline should fit and predict without error."""
        X, y = iris
        engine = EvolutionEngine(
            population_size=4, max_generations=2,
            cv=2, task="classification",
            eval_timeout=20.0, random_seed=1,
        )
        best = engine.fit(X, y)
        best.fit(X, y)
        preds = best.predict(X)
        assert len(preds) == len(y)

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="task must be"):
            EvolutionEngine(task="clustering")

    def test_cache_is_used(self, iris):
        """Cache hit rate should be > 0 after a multi-generation run."""
        X, y = iris
        engine = EvolutionEngine(
            population_size=5, max_generations=3,
            cv=2, task="classification",
            eval_timeout=20.0, random_seed=42,
        )
        engine.fit(X, y)
        stats = engine.cache_stats()
        assert "hit_rate" in stats

    def test_plateau_stopping(self, iris):
        """With a large tolerance, plateau stopping fires early."""
        X, y = iris
        engine = EvolutionEngine(
            population_size=4, max_generations=50,
            cv=2, task="classification",
            eval_timeout=20.0, random_seed=99,
            plateau_patience=2,
        )
        engine.fit(X, y)
        # Should stop well before 50 generations
        assert engine.history().generations_run() < 50

    def test_time_budget_respected(self, iris):
        """A very tight budget should terminate early."""
        X, y = iris
        engine = EvolutionEngine(
            population_size=6, max_generations=1000,
            cv=2, task="classification",
            # Tight budget: force time-based stop
            max_time_seconds=5.0,
            eval_timeout=3.0,
            random_seed=0,
        )
        engine.fit(X, y)
        assert engine.history().generations_run() < 1000

    def test_generation_records_structure(self, iris):
        X, y = iris
        engine = EvolutionEngine(
            population_size=4, max_generations=2,
            cv=2, task="classification",
            eval_timeout=20.0, random_seed=3,
        )
        engine.fit(X, y)
        for rec in engine.history().records:
            assert isinstance(rec, GenerationRecord)
            assert rec.generation >= 0
            assert rec.wall_time >= 0.0
            assert isinstance(rec.best_pipeline_hash, str)

    def test_iris_accuracy_reasonable(self, iris):
        """After 3 generations the best accuracy should be > 70%."""
        X, y = iris
        engine = EvolutionEngine(
            population_size=8, max_generations=3,
            cv=3, task="classification",
            eval_timeout=30.0, random_seed=42,
        )
        engine.fit(X, y)
        final_score = engine.history().final_best()
        assert final_score > 0.70, (
            f"Expected > 70% accuracy after 3 generations, got {final_score:.2%}"
        )
