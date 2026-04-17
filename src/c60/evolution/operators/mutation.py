"""
mutation.py — Mutation operators for pipeline DAG evolution.

Five operators, each implementing the MutationOperator interface:

  NodeReplacementMutation   — swap a node's operation for another of same step_type
  HyperparameterMutation    — perturb one hyperparameter of a random node
  NodeInsertionMutation     — insert a new node on an existing edge
  NodeDeletionMutation      — remove a node, reconnect its predecessor to its successors
  EdgeRedirectionMutation   — redirect one outgoing edge to a different valid target

All operators:
  - Accept a Pipeline and return a NEW Pipeline (never mutate in-place)
  - Manipulate the graph directly (bypassing Pipeline.add_edge type-checks) for
    performance, then return the result for the caller to validate
  - Return a clone of the original if no valid mutation can be found

MutationEngine selects operators by weighted random choice and retries up to
max_attempts times if the result fails validation.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import networkx as nx

from c60.core.pipeline import Pipeline, PipelineStep
from c60.core.registry import OperationRegistry
from c60.core.types import is_compatible
from c60.execution.validator import PipelineValidator


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class MutationOperator(ABC):
    """Abstract base for all mutation operators."""

    @abstractmethod
    def apply(self, pipeline: Pipeline, registry: OperationRegistry) -> Pipeline:
        """
        Return a new (possibly mutated) pipeline.
        Must not modify `pipeline` in-place.
        Returns a clone of `pipeline` if no valid mutation is found.
        """


# ---------------------------------------------------------------------------
# 1. NodeReplacementMutation
# ---------------------------------------------------------------------------

class NodeReplacementMutation(MutationOperator):
    """
    Replace one node's operation with a different one of the same step_type.

    Because step_type maps to a fixed (input_type, output_type) in the registry
    spec, all incident edges automatically remain type-compatible — no edge
    rewiring needed.

    If multiple specs share the same step_type but differ in input/output types
    (e.g. RandomForest takes UnscaledData, LogisticRegression takes ScaledData),
    we filter to only those with matching I/O types to preserve edge validity.
    """

    def apply(self, pipeline: Pipeline, registry: OperationRegistry) -> Pipeline:
        cloned = pipeline.clone()
        nodes = cloned.get_nodes()
        random.shuffle(nodes)

        for step in nodes:
            if step.operation is None:
                continue

            # Find candidate specs: same step_type + same input/output types
            candidates = [
                spec for spec in registry.get_by_type(step.step_type)
                if spec.input_type is step.input_type
                and spec.output_type is step.output_type
                and spec.op_class is not step.operation.__class__
            ]
            if not candidates:
                continue

            spec = random.choice(candidates)
            new_params = spec.sample_params()
            new_op = spec.instantiate(new_params)

            step.name = spec.op_class.__name__
            step.operation = new_op
            step.params = new_params
            return cloned

        return cloned  # no replaceable node found


# ---------------------------------------------------------------------------
# 2. HyperparameterMutation
# ---------------------------------------------------------------------------

class HyperparameterMutation(MutationOperator):
    """
    Perturb one hyperparameter of one randomly selected node.

    The new value is sampled from the spec's search space for that parameter.
    The operation is re-instantiated with the updated params so the change
    takes effect immediately.
    """

    def apply(self, pipeline: Pipeline, registry: OperationRegistry) -> Pipeline:
        cloned = pipeline.clone()
        nodes = cloned.get_nodes()
        random.shuffle(nodes)

        for step in nodes:
            if step.operation is None:
                continue

            spec = registry.get_spec(step.operation.__class__)
            if spec is None or not spec.search_space:
                continue

            # Pick one hyperparameter to mutate
            param_name = random.choice(list(spec.search_space.keys()))
            new_value = spec.search_space[param_name].sample()

            new_params = {**step.params, param_name: new_value}
            new_op = spec.instantiate(new_params)

            step.params = new_params
            step.operation = new_op
            return cloned

        return cloned  # no mutable node found


# ---------------------------------------------------------------------------
# 3. NodeInsertionMutation
# ---------------------------------------------------------------------------

class NodeInsertionMutation(MutationOperator):
    """
    Insert a new node on an existing edge (A → B) → (A → NEW → B).

    The new node is sampled from the registry such that:
      - new.input_type  is compatible with A.output_type
      - new.output_type is compatible with B.input_type

    The original edge is removed; two new edges are added directly on the
    underlying graph (no Pipeline.add_edge validation overhead — the caller
    validates the result).
    """

    def apply(self, pipeline: Pipeline, registry: OperationRegistry) -> Pipeline:
        cloned = pipeline.clone()
        edges = cloned.get_edges()
        if not edges:
            return cloned

        random.shuffle(edges)  # type: ignore[arg-type]

        for from_id, to_id in edges:
            src = cloned.get_step(from_id)
            dst = cloned.get_step(to_id)

            new_step = registry.sample_compatible_step(
                input_type=src.output_type,
                output_type=dst.input_type,
            )
            if new_step is None:
                continue

            # Splice the new node into the edge
            cloned.graph.remove_edge(from_id, to_id)
            cloned.graph.add_node(new_step.id, data=new_step)
            cloned.graph.add_edge(from_id, new_step.id)
            cloned.graph.add_edge(new_step.id, to_id)
            return cloned

        return cloned  # no edge accepted a new node


# ---------------------------------------------------------------------------
# 4. NodeDeletionMutation
# ---------------------------------------------------------------------------

class NodeDeletionMutation(MutationOperator):
    """
    Remove a non-source, non-sink node with exactly one predecessor.

    The predecessor is reconnected directly to all of the deleted node's
    successors.  Type compatibility of each new (predecessor → successor)
    edge is checked before committing.

    We restrict to exactly-one-predecessor nodes for simplicity:
    - Multiple predecessors (fan-in) would require checking all combos
    - Zero predecessors (source) would disconnect the graph
    """

    def apply(self, pipeline: Pipeline, registry: OperationRegistry) -> Pipeline:
        if len(pipeline) <= 2:
            # Nothing to delete — pipeline would become trivially small
            return pipeline.clone()

        cloned = pipeline.clone()
        nodes = cloned.get_nodes()
        random.shuffle(nodes)

        for step in nodes:
            preds = cloned.predecessors(step)
            succs = cloned.successors(step)

            # Must have exactly one predecessor and at least one successor
            if len(preds) != 1 or len(succs) == 0:
                continue

            pred = preds[0]

            # Check type compatibility: pred.output_type → each succ.input_type
            all_compatible = all(
                is_compatible(pred.output_type, s.input_type)
                for s in succs
            )
            if not all_compatible:
                continue

            # Perform deletion: remove step, rewire
            cloned.graph.remove_node(step.id)  # networkx removes incident edges
            for succ in succs:
                cloned.graph.add_edge(pred.id, succ.id)

            return cloned

        return cloned  # no deletable node found


# ---------------------------------------------------------------------------
# 5. EdgeRedirectionMutation
# ---------------------------------------------------------------------------

class EdgeRedirectionMutation(MutationOperator):
    """
    Redirect one outgoing edge (A → B) to a different target C: (A → C).

    Valid redirection requires:
      - (A → C) does not already exist
      - is_compatible(A.output_type, C.input_type)
      - The result is still a DAG (no cycle introduced)

    B is left in the graph.  If B had only A as a predecessor and now has
    no predecessors, it becomes a disconnected source — the validator will
    catch this and the operator will return the original if needed.
    This is intentional: occasionally creating a new branch or disconnecting
    a sub-path drives structural diversity.
    """

    def apply(self, pipeline: Pipeline, registry: OperationRegistry) -> Pipeline:
        cloned = pipeline.clone()
        edges = cloned.get_edges()
        if not edges:
            return cloned

        random.shuffle(edges)  # type: ignore[arg-type]

        for from_id, to_id in edges:
            src = cloned.get_step(from_id)
            other_nodes = [
                s for s in cloned.get_nodes()
                if s.id != from_id
                and s.id != to_id
                and not cloned.graph.has_edge(from_id, s.id)
                and is_compatible(src.output_type, s.input_type)
            ]
            if not other_nodes:
                continue

            target = random.choice(other_nodes)

            # Tentatively add the new edge, check for cycles
            cloned.graph.remove_edge(from_id, to_id)
            cloned.graph.add_edge(from_id, target.id)

            if nx.is_directed_acyclic_graph(cloned.graph):
                return cloned  # success

            # Cycle detected — roll back and try next edge
            cloned.graph.remove_edge(from_id, target.id)
            cloned.graph.add_edge(from_id, to_id)

        return cloned  # no valid redirection found


# ---------------------------------------------------------------------------
# MutationEngine
# ---------------------------------------------------------------------------

class MutationEngine:
    """
    Selects and applies mutation operators by weighted random choice.

    Parameters
    ----------
    registry : OperationRegistry
        Used by operators that need to sample new operations.
    operators : list of MutationOperator, optional
        The operators to choose from.  Defaults to all five operators.
    probabilities : list of float, optional
        Selection weights (must sum to 1 or will be normalised).
        Defaults to [0.30, 0.30, 0.20, 0.10, 0.10].
    max_attempts : int
        How many times to retry before giving up and returning a clone.
    """

    _DEFAULT_OPERATORS: List[MutationOperator] = [
        NodeReplacementMutation(),
        HyperparameterMutation(),
        NodeInsertionMutation(),
        NodeDeletionMutation(),
        EdgeRedirectionMutation(),
    ]
    _DEFAULT_PROBS: List[float] = [0.30, 0.30, 0.20, 0.10, 0.10]

    def __init__(
        self,
        registry: OperationRegistry,
        operators: Optional[List[MutationOperator]] = None,
        probabilities: Optional[List[float]] = None,
        max_attempts: int = 5,
    ) -> None:
        self.registry = registry
        self.operators = operators or list(self._DEFAULT_OPERATORS)
        self.probabilities = probabilities or list(self._DEFAULT_PROBS)
        self.max_attempts = max_attempts
        self._validator = PipelineValidator()

        if len(self.operators) != len(self.probabilities):
            raise ValueError(
                "operators and probabilities must have the same length."
            )

    def mutate(self, pipeline: Pipeline) -> Pipeline:
        """
        Apply a randomly selected mutation operator.

        Retries up to max_attempts times, returning the first valid result.
        Falls back to a clean clone if all attempts produce invalid pipelines.
        """
        for _ in range(self.max_attempts):
            operator = random.choices(
                self.operators, weights=self.probabilities, k=1
            )[0]
            try:
                result = operator.apply(pipeline, self.registry)
                if self._validator.validate(result).is_valid:
                    return result
            except Exception:
                continue

        return pipeline.clone()

    def operator_names(self) -> List[str]:
        return [op.__class__.__name__ for op in self.operators]
