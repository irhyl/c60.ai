"""
parallel.py — Thread-pool parallel population evaluation.

The dominant cost in each GA generation is cross-validated fitness
evaluation: each individual pipeline runs k-fold CV independently of
every other.  ParallelEvaluator exploits this independence by evaluating
the full population concurrently with a ThreadPoolExecutor.

Why threads, not processes?
  - sklearn releases the GIL during most numerical computations (NumPy,
    LAPACK, BLAS) so threads can overlap real CPU work.
  - Processes avoid the GIL but incur process-spawn overhead and large
    pickling cost for the evaluation cache and data arrays.
  - On a 4-core machine, threads typically give 2-3× speedup for
    independent CV folds (GIL contention matters less than I/O and
    pure-C extensions).

Thread safety:
  - Each Individual.evaluate() call is fully independent (no shared
    mutable state beyond the EvaluationCache, which uses a threading.Lock).
  - The population list itself is never modified during evaluation; only
    ind.fitness is written on each individual's own object.

Usage
-----
    from c60.execution.parallel import ParallelEvaluator

    parallel = ParallelEvaluator(n_jobs=4)
    parallel.evaluate_population(population, evaluator, X, y)
    # Equivalent to population.evaluate_all(evaluator, X, y)
    # but may run faster on multi-core hardware.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from c60.evaluation.fitness import FitnessEvaluator
from c60.evolution.population import Individual, Population


class ParallelEvaluator:
    """
    Evaluate a Population's individuals concurrently.

    Parameters
    ----------
    n_jobs : int
        Number of worker threads.  -1 (default) uses os.cpu_count().
        1 falls back to sequential evaluation (useful for debugging).
    """

    def __init__(self, n_jobs: int = -1) -> None:
        if n_jobs == 0:
            raise ValueError("n_jobs must be -1 or a positive integer, not 0.")
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        population: Population,
        evaluator: FitnessEvaluator,
        X,
        y,
        reevaluate: bool = False,
    ) -> None:
        """
        Evaluate all unevaluated (or all, if reevaluate=True) individuals
        in the population, potentially in parallel.

        Results are written in-place to ind.fitness on each Individual.
        The population's internal list is not reordered.

        Parameters
        ----------
        population : Population
        evaluator : FitnessEvaluator
        X, y : array-like
            Training data and labels.
        reevaluate : bool
            If True, re-evaluate even individuals that already have a
            fitness score (ignores cache hits).
        """
        targets: List[Individual] = [
            ind for ind in population
            if ind.fitness is None or reevaluate
        ]

        if not targets:
            return

        workers = self._n_workers(len(targets))

        if workers == 1:
            # Sequential fallback — avoids thread overhead for tiny populations
            for ind in targets:
                ind.evaluate(evaluator, X, y)
            return

        # ------------------------------------------------------------------
        # Thread-parallel evaluation
        # ------------------------------------------------------------------
        # Each future maps back to its Individual so we can store the result.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_ind = {
                pool.submit(_eval_one, ind, evaluator, X, y): ind
                for ind in targets
            }
            for future in as_completed(future_to_ind):
                ind = future_to_ind[future]
                try:
                    ind.fitness = future.result()
                except Exception as exc:
                    # Evaluation failed catastrophically — mark as failed
                    from c60.evaluation.cache import FitnessResult
                    ind.fitness = FitnessResult(
                        raw_score=float("-inf"),
                        adjusted_score=float("-inf"),
                        cv_scores=[],
                        wall_time=0.0,
                        n_nodes=len(ind.pipeline),
                        error=str(exc),
                    )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def effective_n_jobs(self) -> int:
        """Return the actual number of workers that will be used."""
        return self._n_workers(1)

    def _n_workers(self, n_targets: int) -> int:
        cpu = os.cpu_count() or 1
        max_workers = cpu if self.n_jobs == -1 else self.n_jobs
        # No point spawning more threads than there are tasks
        return min(max_workers, n_targets)

    def __repr__(self) -> str:
        return f"ParallelEvaluator(n_jobs={self.n_jobs})"


# ---------------------------------------------------------------------------
# Module-level helper (must be picklable / importable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _eval_one(ind: Individual, evaluator: FitnessEvaluator, X, y):
    """Evaluate a single individual and return its FitnessResult."""
    return evaluator.evaluate(ind.pipeline, X, y)
