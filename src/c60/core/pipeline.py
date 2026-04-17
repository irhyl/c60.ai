"""
pipeline.py — Core pipeline data structures for C60.ai

PipelineStep  — a single typed node in the pipeline DAG
Pipeline      — a typed Directed Acyclic Graph of PipelineSteps

Design principles:
  - Every step carries input_type / output_type from the type lattice.
  - Every edge is validated for type compatibility on insertion.
  - Cycles are rejected immediately with rollback.
  - clone() produces a fully independent deep copy (safe for mutation).
  - to_dict() / from_dict() enables serialization and cache hashing.
  - fit() / predict() / transform() mirror the scikit-learn interface so
    the pipeline can be used as a drop-in estimator.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type

import networkx as nx

from c60.core.types import (
    DataType,
    CleanTabular,
    is_compatible,
    type_name,
)


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------

class PipelineStep:
    """
    A single node in the pipeline DAG.

    Each step wraps a callable operation (e.g. a scikit-learn transformer or
    estimator) together with its metadata: unique id, human name, step type
    label, input/output data types, and hyperparameter configuration.

    The step delegates fit / transform / predict to the underlying operation,
    so PipelineStep can be used transparently wherever the operation would be.

    Attributes
    ----------
    id : str
        UUID4 string.  Uniquely identifies this node within a Pipeline.
    name : str
        Human-readable label used in logs and visualisations.
    step_type : str
        Categorical label from the StepType taxonomy (e.g. "scaler",
        "classifier", "feature_engineer").  Used by genetic operators to
        find compatible replacement operations.
    operation : callable or None
        The actual ML operation object (sklearn transformer / estimator, or
        any object with fit/transform/predict).
    params : dict
        Hyperparameter configuration passed to / stored for the operation.
    input_type : Type[DataType]
        The data type this step accepts as input.
    output_type : Type[DataType]
        The data type this step produces as output.
    """

    def __init__(
        self,
        step_id: Optional[str] = None,
        name: str = "UnnamedStep",
        step_type: str = "generic",
        operation: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        input_type: Type[DataType] = CleanTabular,
        output_type: Type[DataType] = CleanTabular,
    ) -> None:
        self.id: str = step_id if step_id is not None else str(uuid.uuid4())
        self.name: str = name
        self.step_type: str = step_type
        self.operation: Optional[Any] = operation
        self.params: Dict[str, Any] = params if params is not None else {}
        self.input_type: Type[DataType] = input_type
        self.output_type: Type[DataType] = output_type

        # Track whether the underlying operation has been fitted
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # scikit-learn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any = None) -> "PipelineStep":
        """
        Fit the underlying operation.  Returns self for chaining.
        No-op if operation is None.
        """
        if self.operation is not None:
            if y is not None:
                self.operation.fit(X, y)
            else:
                self.operation.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: Any) -> Any:
        """
        Apply the transform method of the underlying operation.
        Falls through to predict() if transform is not available
        (e.g. for terminal estimator nodes).
        """
        if self.operation is None:
            return X
        if hasattr(self.operation, "transform"):
            return self.operation.transform(X)
        if hasattr(self.operation, "predict"):
            return self.operation.predict(X)
        raise AttributeError(
            f"Operation '{self.name}' has neither transform() nor predict()."
        )

    def predict(self, X: Any) -> Any:
        """Apply predict() on the underlying estimator."""
        if self.operation is None:
            return X
        if not hasattr(self.operation, "predict"):
            raise AttributeError(
                f"Operation '{self.name}' has no predict() method."
            )
        return self.operation.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Apply predict_proba() if the operation supports it."""
        if self.operation is None:
            raise AttributeError("No operation set — cannot call predict_proba.")
        if not hasattr(self.operation, "predict_proba"):
            raise AttributeError(
                f"Operation '{self.name}' has no predict_proba() method."
            )
        return self.operation.predict_proba(X)

    # ------------------------------------------------------------------
    # Cloning and serialisation
    # ------------------------------------------------------------------

    def clone(self) -> "PipelineStep":
        """
        Return a deep independent copy of this step.
        The cloned step is unfitted (fresh operation instance).
        """
        cloned_op = copy.deepcopy(self.operation)
        # Reset fitted state on the clone so it must be re-trained
        if cloned_op is not None and hasattr(cloned_op, "__sklearn_is_fitted__"):
            pass  # deepcopy handles internal state

        return PipelineStep(
            step_id=str(uuid.uuid4()),  # new id — it's a distinct node
            name=self.name,
            step_type=self.step_type,
            operation=copy.deepcopy(self.operation),
            params=copy.deepcopy(self.params),
            input_type=self.input_type,
            output_type=self.output_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise to a JSON-compatible dict.
        The operation is represented by its fully-qualified class name and
        params; the actual fitted weights are NOT serialised here (use
        joblib for that).
        """
        op_class = (
            f"{self.operation.__class__.__module__}.{self.operation.__class__.__name__}"
            if self.operation is not None
            else None
        )
        return {
            "id": self.id,
            "name": self.name,
            "step_type": self.step_type,
            "operation_class": op_class,
            "params": self.params,
            "input_type": self.input_type.__name__,
            "output_type": self.output_type.__name__,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PipelineStep):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return (
            f"PipelineStep(id='{self.id[:8]}...', name='{self.name}', "
            f"type='{self.step_type}', "
            f"in={type_name(self.input_type)}, out={type_name(self.output_type)})"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    A typed Directed Acyclic Graph of PipelineSteps.

    Nodes   — PipelineStep instances (stored as networkx node attributes)
    Edges   — directed data-flow connections, validated for:
                1. Both endpoints must exist
                2. Adding the edge must not create a cycle
                3. output_type(source) must be compatible with input_type(dest)

    Execution follows topological order.  Fan-out (one node feeds many) and
    fan-in (many nodes feed one, e.g. an Ensembler) are both supported.

    The Pipeline is the unit of evolution: clone() → mutate → evaluate.
    """

    def __init__(self, name: str = "UnnamedPipeline") -> None:
        self.name: str = name
        self.graph: nx.DiGraph = nx.DiGraph()
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_step(self, step: PipelineStep) -> None:
        """
        Add a PipelineStep as a node.

        Raises
        ------
        ValueError
            If a step with the same id already exists.
        """
        if self.graph.has_node(step.id):
            raise ValueError(
                f"Step with id '{step.id}' already exists in pipeline '{self.name}'. "
                "Each step must have a unique id."
            )
        self.graph.add_node(step.id, data=step)

    def add_edge(self, from_id: str, to_id: str) -> None:
        """
        Add a directed edge representing data flow from `from_id` to `to_id`.

        Validates:
          1. Both node ids exist.
          2. The edge does not create a cycle (rolls back if it does).
          3. output_type of source is compatible with input_type of dest.

        Raises
        ------
        ValueError
            On any validation failure.  The graph is left unchanged.
        """
        if not self.graph.has_node(from_id):
            raise ValueError(
                f"Source step '{from_id}' not found in pipeline '{self.name}'."
            )
        if not self.graph.has_node(to_id):
            raise ValueError(
                f"Destination step '{to_id}' not found in pipeline '{self.name}'."
            )

        # Type compatibility check
        src_step: PipelineStep = self.graph.nodes[from_id]["data"]
        dst_step: PipelineStep = self.graph.nodes[to_id]["data"]
        if not is_compatible(src_step.output_type, dst_step.input_type):
            raise TypeError(
                f"Type mismatch: '{src_step.name}' produces "
                f"'{type_name(src_step.output_type)}' but "
                f"'{dst_step.name}' expects "
                f"'{type_name(dst_step.input_type)}'."
            )

        self.graph.add_edge(from_id, to_id)

        # Cycle check with rollback
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_id, to_id)
            raise ValueError(
                f"Adding edge '{from_id}' → '{to_id}' would create a cycle. "
                "Pipelines must be acyclic."
            )

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def get_step(self, step_id: str) -> PipelineStep:
        """Return the PipelineStep for the given id."""
        try:
            return self.graph.nodes[step_id]["data"]
        except KeyError:
            raise KeyError(
                f"Step with id '{step_id}' not found in pipeline '{self.name}'."
            )

    def get_nodes(self) -> List[PipelineStep]:
        """Return all steps (order not guaranteed to be topological)."""
        return [self.graph.nodes[nid]["data"] for nid in self.graph.nodes]

    def get_edges(self) -> List[Tuple[str, str]]:
        """Return all edges as (from_id, to_id) tuples."""
        return list(self.graph.edges)

    def get_sources(self) -> List[PipelineStep]:
        """Return steps with no incoming edges (pipeline entry points)."""
        return [
            self.graph.nodes[nid]["data"]
            for nid in self.graph.nodes
            if self.graph.in_degree(nid) == 0
        ]

    def get_sinks(self) -> List[PipelineStep]:
        """Return steps with no outgoing edges (pipeline exit points)."""
        return [
            self.graph.nodes[nid]["data"]
            for nid in self.graph.nodes
            if self.graph.out_degree(nid) == 0
        ]

    def predecessors(self, step: PipelineStep) -> List[PipelineStep]:
        """Return the immediate predecessor steps of `step`."""
        return [
            self.graph.nodes[nid]["data"]
            for nid in self.graph.predecessors(step.id)
        ]

    def successors(self, step: PipelineStep) -> List[PipelineStep]:
        """Return the immediate successor steps of `step`."""
        return [
            self.graph.nodes[nid]["data"]
            for nid in self.graph.successors(step.id)
        ]

    def topological_order(self) -> List[PipelineStep]:
        """
        Return all steps in a valid topological execution order.
        Each step appears after all its predecessors.
        """
        return [
            self.graph.nodes[nid]["data"]
            for nid in nx.topological_sort(self.graph)
        ]

    # ------------------------------------------------------------------
    # Execution (scikit-learn compatible)
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any = None) -> "Pipeline":
        """
        Fit the pipeline by executing each step in topological order.

        Transformer steps: fit_transform(X) to produce input for successors.
        Estimator steps (sinks): fit(X, y).

        Fan-out: a step's output is forwarded independently to each successor.
        Fan-in: a step receives a list of outputs from all its predecessors;
                the operation must handle list input (e.g. an Ensembler).
        """
        node_outputs: Dict[str, Any] = {}

        for step in self.topological_order():
            preds = self.predecessors(step)

            if not preds:
                # Source node — receives raw X
                X_in = X
            elif len(preds) == 1:
                X_in = node_outputs[preds[0].id]
            else:
                # Fan-in: collect outputs from all predecessors
                X_in = [node_outputs[p.id] for p in preds]

            succs = self.successors(step)
            is_sink = len(succs) == 0

            if is_sink:
                step.fit(X_in, y)
                node_outputs[step.id] = None  # sinks don't forward data
            else:
                step.fit(X_in, y)
                node_outputs[step.id] = step.transform(X_in)

        self._is_fitted = True
        return self

    def predict(self, X: Any) -> Any:
        """
        Run prediction through the fitted pipeline.
        Returns the output of the sink node.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline has not been fitted. Call fit() before predict()."
            )

        node_outputs: Dict[str, Any] = {}

        for step in self.topological_order():
            preds = self.predecessors(step)

            if not preds:
                X_in = X
            elif len(preds) == 1:
                X_in = node_outputs[preds[0].id]
            else:
                X_in = [node_outputs[p.id] for p in preds]

            succs = self.successors(step)
            is_sink = len(succs) == 0

            if is_sink:
                node_outputs[step.id] = step.predict(X_in)
            else:
                node_outputs[step.id] = step.transform(X_in)

        # Return the output of the first (and ideally only) sink
        sinks = self.get_sinks()
        return node_outputs[sinks[0].id]

    def transform(self, X: Any) -> Any:
        """
        Run the pipeline as a pure transformer (no predict at the sink).
        Useful when the pipeline is used as a preprocessing sub-graph.
        """
        node_outputs: Dict[str, Any] = {}

        for step in self.topological_order():
            preds = self.predecessors(step)

            if not preds:
                X_in = X
            elif len(preds) == 1:
                X_in = node_outputs[preds[0].id]
            else:
                X_in = [node_outputs[p.id] for p in preds]

            node_outputs[step.id] = step.transform(X_in)

        sinks = self.get_sinks()
        return node_outputs[sinks[0].id]

    # ------------------------------------------------------------------
    # Cloning and serialisation
    # ------------------------------------------------------------------

    def clone(self) -> "Pipeline":
        """
        Return a deep independent copy of this pipeline.
        All steps are cloned (new ids, unfitted).  The graph topology
        is preserved with the new ids.
        """
        new_pipeline = Pipeline(name=self.name)

        # Build a mapping from old step id → new cloned step
        id_map: Dict[str, str] = {}
        for step in self.get_nodes():
            new_step = step.clone()
            id_map[step.id] = new_step.id
            new_pipeline.add_step(new_step)

        # Recreate edges using new ids (skip type check — same topology)
        for from_id, to_id in self.get_edges():
            new_pipeline.graph.add_edge(id_map[from_id], id_map[to_id])

        return new_pipeline

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the pipeline structure to a JSON-compatible dict."""
        return {
            "name": self.name,
            "steps": [step.to_dict() for step in self.get_nodes()],
            "edges": list(self.get_edges()),
        }

    def structure_hash(self) -> str:
        """
        Return a deterministic hash of the pipeline structure.
        Used by EvaluationCache to detect duplicate pipelines.
        Hash is based on the canonical representation of nodes + edges,
        independent of step ids (which change on clone).
        """
        import hashlib
        import json

        steps_repr = sorted(
            [
                (s.step_type, s.operation.__class__.__name__ if s.operation else "None",
                 json.dumps(s.params, sort_keys=True))
                for s in self.get_nodes()
            ]
        )
        # Represent topology by step_type sequences along each edge
        edges_repr = sorted(
            [
                (
                    self.get_step(f).step_type,
                    self.get_step(t).step_type,
                )
                for f, t in self.get_edges()
            ]
        )
        payload = json.dumps({"steps": steps_repr, "edges": edges_repr}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    def __repr__(self) -> str:
        return (
            f"Pipeline(name='{self.name}', steps={len(self)}, "
            f"edges={self.graph.number_of_edges()})"
        )

    def __str__(self) -> str:
        header = f"Pipeline: {self.name}"
        try:
            order = self.topological_order()
            steps_str = " → ".join(
                f"{s.name}({type_name(s.output_type)})" for s in order
            )
        except Exception:
            steps_str = ", ".join(s.name for s in self.get_nodes())
        edges_str = (
            str(self.get_edges()) if self.get_edges() else "none"
        )
        return f"{header}\n  Flow : {steps_str}\n  Edges: {edges_str}"
