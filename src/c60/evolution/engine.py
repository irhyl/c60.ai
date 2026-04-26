"""
engine.py — EvolutionEngine: the main genetic algorithm loop.

EvolutionEngine ties together every component built so far:

  Population        — N pipeline individuals
  FitnessEvaluator  — cross-validation scorer with caching
  TournamentSelector — parent selection
  ElitismPreserver  — carry top-k unchanged each generation
  MutationEngine    — 5 structural / parametric mutation operators
  SubgraphExchangeCrossover — typed subgraph recombination

One call to engine.fit(X, y) runs the full GA and returns the best
Pipeline found, along with a GenerationLog for introspection.

GA Loop (per generation)
------------------------
1.  Evaluate all unevaluated individuals (uses cache to skip repeats)
2.  Log generation statistics
3.  Check stopping criteria (max generations, time budget, fitness plateau)
4.  Preserve elites → next generation
5.  While next_gen < population_size:
      a. Select two parents via tournament
      b. Crossover with probability p_crossover
      c. Mutate each offspring with probability p_mutation
      d. Append to next_gen
6.  Replace population with next_gen
7.  Go to 1

Design notes
------------
- The engine never raises on a broken pipeline; FitnessEvaluator handles
  that by returning float('-inf').
- Stopping criteria are checked after each generation:
    * max_generations reached
    * wall-clock time budget exceeded
    * best fitness has not improved by > plateau_tolerance for
      plateau_patience consecutive generations
- The EvolutionLog stores one GenerationRecord per generation and can
  be used by PipelineIntrospector (Phase 6) to trace lineage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from c60.core.pipeline import Pipeline
from c60.core.registry import OperationRegistry, default_registry
from c60.evaluation.cache import EvaluationCache
from c60.evaluation.fitness import FitnessEvaluator
from c60.evolution.operators.crossover import SubgraphExchangeCrossover
from c60.evolution.operators.mutation import (
    HyperparameterMutation,
    MutationEngine,
)
from c60.evolution.population import Individual, Population
from c60.evolution.selection import ElitismPreserver, TournamentSelector


# ---------------------------------------------------------------------------
# Generation log
# ---------------------------------------------------------------------------

@dataclass
class GenerationRecord:
    """Statistics for one completed generation."""
    generation: int
    best_score: float
    mean_score: float
    population_size: int
    wall_time: float          # cumulative seconds since engine.fit() was called
    best_pipeline_hash: str


@dataclass
class EvolutionLog:
    """Complete history of the evolution run."""
    records: List[GenerationRecord] = field(default_factory=list)

    def append(self, record: GenerationRecord) -> None:
        self.records.append(record)

    def best_scores(self) -> List[float]:
        return [r.best_score for r in self.records]

    def mean_scores(self) -> List[float]:
        return [r.mean_score for r in self.records]

    def final_best(self) -> float:
        return self.records[-1].best_score if self.records else float("-inf")

    def generations_run(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return (
            f"EvolutionLog("
            f"generations={self.generations_run()}, "
            f"best={self.final_best():.4f})"
        )


# ---------------------------------------------------------------------------
# EvolutionEngine
# ---------------------------------------------------------------------------

class EvolutionEngine:
    """
    Genetic algorithm engine for pipeline optimisation.

    Parameters
    ----------
    population_size : int
        Number of pipeline individuals per generation.
    max_generations : int
        Hard upper limit on generations.
    max_time_seconds : float
        Wall-clock budget.  Evolution stops when exceeded (after the
        current generation completes evaluation).
    crossover_rate : float
        Probability that two selected parents undergo crossover.
        Otherwise each parent is cloned as-is.
    mutation_rate : float
        Per-offspring probability that a mutation operator is applied.
    elite_count : int
        Number of top individuals carried unchanged to each new generation.
    tournament_size : int
        Number of contestants per parent selection tournament.
    plateau_patience : int
        Stop if best fitness does not improve by more than plateau_tolerance
        for this many consecutive generations.
    plateau_tolerance : float
        Minimum improvement in best score to reset the plateau counter.
    metric : str
        Fitness metric (e.g. "accuracy", "r2").
    task : str
        "classification" or "regression".
    cv : int
        Number of cross-validation folds per fitness evaluation.
    eval_timeout : float
        Per-pipeline evaluation timeout in seconds.
    complexity_penalty : float
        Parsimony pressure coefficient λ.
    structural_search : bool
        If True (default), all five genetic operators are active including
        graph-structural mutations (node insertion/deletion/replacement,
        edge redirection) and subgraph crossover.
        If False, only hyperparameter perturbation is applied and crossover
        is disabled — producing a population-based HPO baseline that
        searches the same topology space as fixed-template AutoML.
    registry : OperationRegistry, optional
        Defaults to default_registry.
    random_seed : int, optional
        Seeds Python's random module for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 20,
        max_generations: int = 30,
        max_time_seconds: float = 600.0,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.9,
        elite_count: int = 2,
        tournament_size: int = 3,
        plateau_patience: int = 8,
        plateau_tolerance: float = 1e-4,
        metric: Optional[str] = None,
        task: str = "classification",
        cv: int = 3,
        eval_timeout: float = 60.0,
        complexity_penalty: float = 0.005,
        structural_search: bool = True,
        registry: Optional[OperationRegistry] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got '{task}'."
            )

        self.population_size = population_size
        self.max_generations = max_generations
        self.max_time_seconds = max_time_seconds
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.plateau_patience = plateau_patience
        self._plateau_tolerance = plateau_tolerance
        self.task = task
        self.structural_search = structural_search
        self.registry = registry or default_registry

        if random_seed is not None:
            import random as _random
            _random.seed(random_seed)

        # Sub-components
        self._cache = EvaluationCache(max_size=2000)
        self._evaluator = FitnessEvaluator(
            metric=metric,
            cv=cv,
            task=task,
            timeout_seconds=eval_timeout,
            complexity_penalty=complexity_penalty,
            cache=self._cache,
        )
        self._selector = TournamentSelector(tournament_size=tournament_size)
        self._elitism = ElitismPreserver(elite_count=elite_count)

        if structural_search:
            self._mutator = MutationEngine(registry=self.registry)
            self._crossover: Optional[SubgraphExchangeCrossover] = (
                SubgraphExchangeCrossover()
            )
        else:
            # HPO-only mode: topology is fixed, only hyperparameters evolve
            self._mutator = MutationEngine(
                registry=self.registry,
                operators=[HyperparameterMutation()],
                probabilities=[1.0],
            )
            self._crossover = None

        self._log = EvolutionLog()
        self._best_pipeline: Optional[Pipeline] = None
        self._population: Optional[Population] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any) -> Pipeline:
        """
        Run the genetic algorithm on (X, y).

        Returns the best pipeline found across all generations.
        """
        t_start = time.perf_counter()
        self._log = EvolutionLog()

        # --- Initialise population ---
        pop = Population(self.population_size)
        pop.initialize(self.registry, task=self.task)
        self._population = pop

        plateau_counter = 0
        prev_best = float("-inf")

        for generation in range(self.max_generations):

            # --- Evaluate ---
            pop.evaluate_all(self._evaluator, X, y)

            best_ind = pop.best()
            mean = pop.mean_score()
            elapsed = time.perf_counter() - t_start

            # --- Log ---
            self._log.append(GenerationRecord(
                generation=generation,
                best_score=best_ind.score,
                mean_score=mean,
                population_size=len(pop),
                wall_time=elapsed,
                best_pipeline_hash=best_ind.pipeline.structure_hash(),
            ))

            # --- Track global best ---
            if (
                self._best_pipeline is None
                or best_ind.score > self._log.records[-2].best_score
                if len(self._log.records) >= 2 else True
            ):
                self._best_pipeline = best_ind.pipeline.clone()

            # --- Stopping criteria ---
            improvement = best_ind.score - prev_best
            if improvement > self.plateau_tolerance:
                plateau_counter = 0
            else:
                plateau_counter += 1

            prev_best = best_ind.score

            stop_reason = None
            if generation >= self.max_generations - 1:
                stop_reason = "max_generations"
            elif elapsed >= self.max_time_seconds:
                stop_reason = "time_budget"
            elif plateau_counter >= self.plateau_patience:
                stop_reason = "plateau"

            if stop_reason and generation < self.max_generations - 1:
                break

            # --- Build next generation ---
            next_gen_inds: List[Individual] = []

            # Elites carry over unchanged (with their known fitness scores)
            next_gen_inds.extend(self._elitism.get_elites(pop))

            # Fill the rest with offspring
            while len(next_gen_inds) < self.population_size:
                p1, p2 = self._selector.select_pair(pop)

                # Crossover (disabled in HPO-only mode)
                import random as _rand
                if (
                    self._crossover is not None
                    and _rand.random() < self.crossover_rate
                ):
                    child1_pipe, child2_pipe = self._crossover.apply(
                        p1.pipeline, p2.pipeline
                    )
                else:
                    child1_pipe = p1.pipeline.clone()
                    child2_pipe = p2.pipeline.clone()

                # Mutation
                if _rand.random() < self.mutation_rate:
                    child1_pipe = self._mutator.mutate(child1_pipe)
                if _rand.random() < self.mutation_rate:
                    child2_pipe = self._mutator.mutate(child2_pipe)

                next_gen_inds.append(Individual(pipeline=child1_pipe))
                if len(next_gen_inds) < self.population_size:
                    next_gen_inds.append(Individual(pipeline=child2_pipe))

            # Replace population
            pop._individuals = next_gen_inds[: self.population_size]

        # Final evaluation pass to make sure we have the true best
        pop.evaluate_all(self._evaluator, X, y)
        final_best = pop.best()
        if final_best.score > (self._best_pipeline and
                               self._evaluator.evaluate(
                                   self._best_pipeline, X, y).adjusted_score
                               or float("-inf")):
            self._best_pipeline = final_best.pipeline.clone()

        return self._best_pipeline

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def plateau_tolerance(self) -> float:
        return self._plateau_tolerance

    def history(self) -> EvolutionLog:
        """Return the evolution log from the last fit() call."""
        return self._log

    def best_pipeline(self) -> Optional[Pipeline]:
        """Return the best pipeline found so far (None before fit() is called)."""
        return self._best_pipeline

    def cache_stats(self) -> dict:
        """Return a dict of evaluation cache statistics."""
        c = self._cache
        return {
            "hits": c.hits,
            "misses": c.misses,
            "hit_rate": c.hit_rate,
            "size": c.size,
        }

    def __repr__(self) -> str:
        mode = "full" if self.structural_search else "hpo-only"
        return (
            f"EvolutionEngine("
            f"pop={self.population_size}, "
            f"max_gen={self.max_generations}, "
            f"task={self.task}, "
            f"mode={mode})"
        )
