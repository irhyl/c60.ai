"""
population.py — Population management for the C60.ai evolution engine.

Individual  — one pipeline candidate + its fitness result
Population  — the full set of N individuals at any generation

Population handles:
  - Seeded initialization from the OperationRegistry
  - Bulk fitness evaluation (with optional parallelism hook)
  - Querying best / sorted individuals
  - Generating a diverse initial set using type-guided random DAG construction
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from c60.core.pipeline import Pipeline, PipelineStep
from c60.core.registry import OperationRegistry
from c60.core.types import (
    CleanTabular, ScaledData, ClassLabels, RegressionValues,
    UnscaledData, Features,
)
from c60.evaluation.cache import FitnessResult
from c60.evaluation.fitness import FitnessEvaluator


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """
    One member of the population: a pipeline and its evaluated fitness.

    fitness is None until evaluate() has been called.
    """
    pipeline: Pipeline
    fitness: Optional[FitnessResult] = field(default=None, compare=False)

    @property
    def score(self) -> float:
        """
        Adjusted fitness score used for selection.
        Returns float('-inf') if not yet evaluated or if evaluation failed.
        """
        if self.fitness is None:
            return float("-inf")
        return self.fitness.adjusted_score

    def evaluate(self, evaluator: FitnessEvaluator, X, y) -> None:
        """Evaluate this individual in-place."""
        self.fitness = evaluator.evaluate(self.pipeline, X, y)

    def __repr__(self) -> str:
        score_str = f"{self.score:.4f}" if self.fitness else "unevaluated"
        return f"Individual(score={score_str}, pipeline={self.pipeline!r})"


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

class Population:
    """
    A generation of pipeline candidates.

    Parameters
    ----------
    size : int
        Number of individuals to maintain.
    """

    def __init__(self, size: int) -> None:
        if size < 2:
            raise ValueError(f"Population size must be >= 2, got {size}.")
        self.size = size
        self._individuals: List[Individual] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(
        self,
        registry: OperationRegistry,
        task: str = "classification",
        seed_templates: bool = True,
    ) -> None:
        """
        Generate a diverse initial population using type-guided random DAG
        construction, optionally seeded with known-good template pipelines.

        Parameters
        ----------
        registry : OperationRegistry
        task : "classification" or "regression"
        seed_templates : bool
            If True, include a handful of simple template pipelines at the
            start of the population before filling the rest with random DAGs.
            Templates provide genetic anchors that prevent the population from
            starting entirely lost.
        """
        self._individuals = []

        if seed_templates:
            templates = _build_templates(registry, task)
            for p in templates[: self.size]:
                self._individuals.append(Individual(pipeline=p))

        # Fill the rest with random type-guided DAGs
        attempts = 0
        max_attempts = self.size * 20
        while len(self._individuals) < self.size and attempts < max_attempts:
            attempts += 1
            try:
                p = _random_pipeline(registry, task)
                if p is not None and len(p) >= 2:
                    self._individuals.append(Individual(pipeline=p))
            except Exception:
                continue

        # Trim to exact size (templates might have pushed us over)
        self._individuals = self._individuals[: self.size]

        if len(self._individuals) < self.size:
            # Fallback: duplicate templates if random generation keeps failing
            while len(self._individuals) < self.size:
                base = random.choice(self._individuals)
                self._individuals.append(Individual(pipeline=base.pipeline.clone()))

    def initialize_from(self, pipelines: List[Pipeline]) -> None:
        """Initialize directly from a list of Pipeline objects (used in tests)."""
        self._individuals = [Individual(pipeline=p) for p in pipelines]
        self.size = len(pipelines)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(
        self, evaluator: FitnessEvaluator, X, y, reevaluate: bool = False
    ) -> None:
        """
        Evaluate every unevaluated individual (or all if reevaluate=True).
        """
        for ind in self._individuals:
            if ind.fitness is None or reevaluate:
                ind.evaluate(evaluator, X, y)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def best(self) -> Individual:
        """Return the individual with the highest adjusted score."""
        return max(self._individuals, key=lambda ind: ind.score)

    def sorted_individuals(self) -> List[Individual]:
        """Return all individuals sorted by score descending."""
        return sorted(self._individuals, key=lambda ind: ind.score, reverse=True)

    def top_k(self, k: int) -> List[Individual]:
        """Return the top-k individuals by score."""
        return self.sorted_individuals()[:k]

    def mean_score(self) -> float:
        """Mean adjusted score across evaluated individuals."""
        scores = [
            ind.score for ind in self._individuals
            if ind.fitness is not None and not ind.fitness.failed
        ]
        return sum(scores) / len(scores) if scores else float("-inf")

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self):
        return iter(self._individuals)

    def __getitem__(self, idx: int) -> Individual:
        return self._individuals[idx]

    def __repr__(self) -> str:
        evaluated = sum(1 for ind in self._individuals if ind.fitness is not None)
        return (
            f"Population(size={len(self._individuals)}, "
            f"evaluated={evaluated}, "
            f"best_score={self.best().score:.4f if evaluated else 'n/a'})"
        )


# ---------------------------------------------------------------------------
# Private: pipeline construction helpers
# ---------------------------------------------------------------------------

# Step type sequences for each task (defines the "shape" vocabulary)
_CLF_CHAINS = [
    ["scaler", "classifier"],
    ["scaler", "classifier"],
    ["scaler", "dim_reducer", "classifier"],
    ["scaler", "feature_selector", "classifier"],
    ["scaler", "feature_engineer", "classifier"],
]

_REG_CHAINS = [
    ["scaler", "regressor"],
    ["scaler", "regressor"],
    ["scaler", "dim_reducer", "regressor"],
    ["scaler", "feature_selector_regression", "regressor"],
]


def _make_scaled_clf_template(
    registry: OperationRegistry,
    scaler_name: str,
    clf_name: str,
    warm_params: dict,
) -> Optional[Pipeline]:
    """
    Build a two-step Scaler → Classifier pipeline with specific hyperparameters.

    Uses the registry to find the right specs (so default_params are applied
    correctly) and stores only the search-space params in PipelineStep.params
    so HyperparameterMutation can evolve them normally after seeding.
    """
    scaler_specs = [s for s in registry.get_by_type("scaler")
                    if s.op_class.__name__ == scaler_name]
    clf_specs = [s for s in registry.get_by_type("classifier")
                 if s.op_class.__name__ == clf_name
                 and s.input_type is ScaledData]
    if not scaler_specs or not clf_specs:
        return None

    scaler_spec = scaler_specs[0]
    clf_spec = clf_specs[0]

    scaler_op = scaler_spec.instantiate({})
    scaler_step = PipelineStep(
        name=scaler_spec.op_class.__name__,
        step_type=scaler_spec.step_type,
        operation=scaler_op,
        params={},
        input_type=scaler_spec.input_type,
        output_type=scaler_spec.output_type,
    )

    clf_op = clf_spec.instantiate(warm_params)
    clf_step = PipelineStep(
        name=clf_spec.op_class.__name__,
        step_type=clf_spec.step_type,
        operation=clf_op,
        params=warm_params,
        input_type=clf_spec.input_type,
        output_type=clf_spec.output_type,
    )

    if not _compat(scaler_spec.output_type, clf_spec.input_type):
        return None

    p = Pipeline(name="Individual")
    p.add_step(scaler_step)
    p.add_step(clf_step)
    p.graph.add_edge(scaler_step.id, clf_step.id)
    return p


def _build_templates(
    registry: OperationRegistry, task: str
) -> List[Pipeline]:
    """
    Build simple linear pipeline templates that serve as genetic anchors.

    Always prepends two warm-start seeds (SVC-RBF and LR-L2) so the initial
    population is guaranteed to contain known-strong baselines for scaled data.
    The rest are type-guided random chains from _CLF_CHAINS / _REG_CHAINS.
    """
    pipelines = []

    if task == "classification":
        # Warm-start: StandardScaler → SVC(rbf)  and  StandardScaler → LR(l2)
        for clf_name, warm_params in [
            ("SVC", {"kernel": "rbf", "C": 1.0, "gamma": "scale"}),
            ("LogisticRegression", {"C": 1.0, "max_iter": 500}),
        ]:
            p = _make_scaled_clf_template(registry, "StandardScaler", clf_name, warm_params)
            if p is not None:
                pipelines.append(p)

    chains = _CLF_CHAINS if task == "classification" else _REG_CHAINS
    for chain in chains:
        p = _chain_to_pipeline(registry, chain)
        if p is not None:
            pipelines.append(p)
    return pipelines


def _chain_to_pipeline(
    registry: OperationRegistry, step_types: List[str]
) -> Optional[Pipeline]:
    """Build a linear pipeline from a sequence of step_type labels."""
    steps = []
    for st in step_types:
        specs = registry.get_by_type(st)
        if not specs:
            return None
        spec = random.choice(specs)
        params = spec.sample_params()
        op = spec.instantiate(params)
        step = PipelineStep(
            name=spec.op_class.__name__,
            step_type=st,
            operation=op,
            params=params,
            input_type=spec.input_type,
            output_type=spec.output_type,
        )
        steps.append(step)

    p = Pipeline(name="Individual")
    for step in steps:
        p.add_step(step)

    # Wire edges — skip if types are incompatible (shouldn't happen with our
    # curated chains, but guard anyway)
    from c60.core.types import is_compatible
    for i in range(len(steps) - 1):
        src, dst = steps[i], steps[i + 1]
        if not is_compatible(src.output_type, dst.input_type):
            return None
        p.graph.add_edge(src.id, dst.id)

    return p


def _random_pipeline(
    registry: OperationRegistry, task: str, max_depth: int = 4
) -> Optional[Pipeline]:
    """
    Construct a random type-guided linear pipeline of 2 to max_depth steps.

    Strategy:
      1. Start with a random scaler (produces ScaledData)
      2. Optionally insert a random intermediate step (feature eng / selection)
      3. End with a classifier or regressor

    More complex topologies (branching, ensembling) are left to the
    mutation operators to discover from these linear seeds.
    """
    task_finals = ["classifier"] if task == "classification" else ["regressor"]
    depth = random.randint(2, max_depth)

    # Always start with a scaler
    scaler_step = _sample_step(registry, "scaler")
    if scaler_step is None:
        return None

    steps = [scaler_step]
    current_output = scaler_step.output_type

    # Intermediate steps (optional)
    intermediate_types = ["dim_reducer", "feature_selector", "feature_engineer"]
    for _ in range(depth - 2):
        st = random.choice(intermediate_types)
        specs = [
            s for s in registry.get_by_type(st)
            if _compat(current_output, s.input_type)
        ]
        if not specs:
            break
        spec = random.choice(specs)
        params = spec.sample_params()
        step = PipelineStep(
            name=spec.op_class.__name__,
            step_type=st,
            operation=spec.instantiate(params),
            params=params,
            input_type=spec.input_type,
            output_type=spec.output_type,
        )
        steps.append(step)
        current_output = step.output_type

    # Final step: classifier or regressor
    final_type = random.choice(task_finals)
    specs = [
        s for s in registry.get_by_type(final_type)
        if _compat(current_output, s.input_type)
    ]
    if not specs:
        return None
    spec = random.choice(specs)
    params = spec.sample_params()
    final = PipelineStep(
        name=spec.op_class.__name__,
        step_type=final_type,
        operation=spec.instantiate(params),
        params=params,
        input_type=spec.input_type,
        output_type=spec.output_type,
    )
    steps.append(final)

    # Assemble pipeline
    p = Pipeline(name="Individual")
    for step in steps:
        p.add_step(step)
    for i in range(len(steps) - 1):
        p.graph.add_edge(steps[i].id, steps[i + 1].id)

    return p


def _sample_step(
    registry: OperationRegistry, step_type: str
) -> Optional[PipelineStep]:
    specs = registry.get_by_type(step_type)
    if not specs:
        return None
    spec = random.choice(specs)
    params = spec.sample_params()
    return PipelineStep(
        name=spec.op_class.__name__,
        step_type=step_type,
        operation=spec.instantiate(params),
        params=params,
        input_type=spec.input_type,
        output_type=spec.output_type,
    )


def _compat(out_type, in_type) -> bool:
    from c60.core.types import is_compatible
    return is_compatible(out_type, in_type)
