"""
cache.py — Evaluation cache for pipeline fitness scores.

During evolution, the GA frequently generates structurally identical pipelines
(same operation sequence and hyperparameters, different node UUIDs).  Re-running
cross-validation on a pipeline we have already scored wastes the largest single
cost in the system.

EvaluationCache maps pipeline structure hashes → FitnessResult.
The hash is computed by Pipeline.structure_hash(), which is:
  - independent of node UUIDs (changes on clone)
  - dependent on operation classes + hyperparameters + edge topology

The cache is in-memory per evolution run.  It is NOT persisted across runs
(that is a Phase 8 concern — a SQLite-backed meta-knowledge store).

Statistics are tracked so the caller can measure cache effectiveness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from c60.core.pipeline import Pipeline


# ---------------------------------------------------------------------------
# FitnessResult — the value stored in the cache
# ---------------------------------------------------------------------------

@dataclass
class FitnessResult:
    """
    Complete record of one pipeline evaluation.

    Attributes
    ----------
    raw_score : float
        Mean cross-validation score on the primary metric.
    adjusted_score : float
        raw_score minus the parsimony penalty (complexity * lambda).
        This is the value used for selection.
    cv_scores : list of float
        Individual fold scores (length == cv).
    wall_time : float
        Seconds elapsed during evaluation.
    n_nodes : int
        Number of steps in the pipeline (for parsimony tracking).
    error : str or None
        If evaluation failed, a short description of the exception.
        adjusted_score will be float('-inf') in that case.
    """
    raw_score: float
    adjusted_score: float
    cv_scores: List[float]
    wall_time: float
    n_nodes: int
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def __repr__(self) -> str:
        if self.failed:
            return f"FitnessResult(FAILED: {self.error})"
        return (
            f"FitnessResult("
            f"raw={self.raw_score:.4f}, "
            f"adj={self.adjusted_score:.4f}, "
            f"nodes={self.n_nodes}, "
            f"time={self.wall_time:.1f}s)"
        )


# ---------------------------------------------------------------------------
# EvaluationCache
# ---------------------------------------------------------------------------

class EvaluationCache:
    """
    In-memory LRU-style cache: pipeline structure hash → FitnessResult.

    Parameters
    ----------
    max_size : int
        Maximum number of entries.  When full, the oldest entry is evicted
        (insertion-order FIFO — good enough for our use case; true LRU
        would require an OrderedDict with move-to-end on get).
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._store: Dict[str, FitnessResult] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, pipeline: Pipeline) -> Optional[FitnessResult]:
        """
        Return the cached FitnessResult for this pipeline structure, or None.
        """
        key = pipeline.structure_hash()
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, pipeline: Pipeline, result: FitnessResult) -> None:
        """
        Store a FitnessResult for this pipeline structure.
        Evicts the oldest entry if the cache is at capacity.
        """
        key = pipeline.structure_hash()
        if key in self._store:
            return  # already present — no update needed
        if len(self._store) >= self._max_size:
            # Evict oldest (first inserted)
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[key] = result

    def contains(self, pipeline: Pipeline) -> bool:
        """Return True if this pipeline structure has been evaluated."""
        return pipeline.structure_hash() in self._store

    def clear(self) -> None:
        """Empty the cache and reset statistics."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> str:
        return (
            f"EvaluationCache("
            f"size={self.size}/{self._max_size}, "
            f"hits={self._hits}, "
            f"misses={self._misses}, "
            f"hit_rate={self.hit_rate:.1%})"
        )

    def __repr__(self) -> str:
        return self.stats()
