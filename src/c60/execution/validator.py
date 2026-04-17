"""
validator.py — Structural and type validation for Pipeline objects.

PipelineValidator separates the concern of "is this pipeline valid to run?"
from the Pipeline class itself.  It is called:
  - By the genetic operators before accepting a mutated pipeline
  - By FitnessEvaluator before attempting cross-validation
  - Directly by user code if desired

A pipeline is considered valid if ALL of the following hold:
  1. It is a DAG (no cycles) — guaranteed by Pipeline.add_edge, but re-checked
     here after clone/mutation operations that bypass add_edge
  2. It has at least one source (entry point) and at least one sink (exit point)
  3. Every sink is reachable from every source (no disconnected islands)
  4. Every edge is type-compatible: output_type(A) IS-A input_type(B)
  5. Every sink step has a predict() method (i.e. it is an estimator, not a
     transformer left dangling at the end)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import networkx as nx

from c60.core.pipeline import Pipeline, PipelineStep
from c60.core.types import is_compatible


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Returned by PipelineValidator.validate().

    Attributes
    ----------
    is_valid : bool
        True iff there are no errors.
    errors : list of str
        Human-readable descriptions of every problem found.
    warnings : list of str
        Non-fatal issues (e.g. very large pipeline — may be slow).
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        lines = [f"ValidationResult(valid={self.is_valid})"]
        for e in self.errors:
            lines.append(f"  ERROR   : {e}")
        for w in self.warnings:
            lines.append(f"  WARNING : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class PipelineValidator:
    """
    Validates a Pipeline against a set of structural and semantic rules.

    Usage
    -----
    validator = PipelineValidator()
    result = validator.validate(pipeline)
    if not result.is_valid:
        print(result)
    """

    # Pipelines larger than this trigger a warning (may be slow to evaluate)
    LARGE_PIPELINE_THRESHOLD = 12

    def validate(self, pipeline: Pipeline) -> ValidationResult:
        """
        Run all checks.  Returns a ValidationResult with accumulated errors.
        """
        result = ValidationResult()
        self._check_not_empty(pipeline, result)
        self._check_is_dag(pipeline, result)
        self._check_sources_and_sinks(pipeline, result)
        self._check_connectivity(pipeline, result)
        self._check_type_compatibility(pipeline, result)
        self._check_sink_has_predict(pipeline, result)
        self._check_size_warning(pipeline, result)
        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_not_empty(self, pipeline: Pipeline, result: ValidationResult) -> None:
        if len(pipeline) == 0:
            result.add_error("Pipeline has no steps.")

    def _check_is_dag(self, pipeline: Pipeline, result: ValidationResult) -> None:
        if not nx.is_directed_acyclic_graph(pipeline.graph):
            result.add_error("Pipeline graph contains a cycle — it is not a DAG.")

    def _check_sources_and_sinks(
        self, pipeline: Pipeline, result: ValidationResult
    ) -> None:
        sources = pipeline.get_sources()
        sinks = pipeline.get_sinks()
        if not sources:
            result.add_error("Pipeline has no source nodes (no step with in-degree 0).")
        if not sinks:
            result.add_error("Pipeline has no sink nodes (no step with out-degree 0).")

    def _check_connectivity(
        self, pipeline: Pipeline, result: ValidationResult
    ) -> None:
        """
        Every sink must be reachable from every source.
        Disconnected islands waste compute: unreachable nodes are evaluated
        but contribute nothing to the output.
        """
        sources = pipeline.get_sources()
        sinks = pipeline.get_sinks()
        if not sources or not sinks:
            return  # already reported above

        for source in sources:
            for sink in sinks:
                if not nx.has_path(pipeline.graph, source.id, sink.id):
                    result.add_error(
                        f"Sink '{sink.name}' is not reachable from "
                        f"source '{source.name}'. Pipeline has disconnected islands."
                    )

    def _check_type_compatibility(
        self, pipeline: Pipeline, result: ValidationResult
    ) -> None:
        """
        For every edge (A → B), output_type(A) must be compatible with input_type(B).
        Pipeline.add_edge enforces this, but clone() + direct graph manipulation
        can bypass it, so we re-check here.
        """
        for from_id, to_id in pipeline.get_edges():
            src: PipelineStep = pipeline.get_step(from_id)
            dst: PipelineStep = pipeline.get_step(to_id)
            if not is_compatible(src.output_type, dst.input_type):
                from c60.core.types import type_name
                result.add_error(
                    f"Type mismatch on edge '{src.name}' → '{dst.name}': "
                    f"{type_name(src.output_type)} is not compatible with "
                    f"{type_name(dst.input_type)}."
                )

    def _check_sink_has_predict(
        self, pipeline: Pipeline, result: ValidationResult
    ) -> None:
        """
        Sink nodes must be estimators (have predict()).
        A transformer at the sink means the pipeline produces features,
        not predictions — valid for sub-graphs but not for a full AutoML pipeline.
        """
        for sink in pipeline.get_sinks():
            op = sink.operation
            if op is not None and not hasattr(op, "predict"):
                result.add_warning(
                    f"Sink node '{sink.name}' has no predict() method. "
                    "The pipeline produces transformed features, not predictions. "
                    "This is valid only if used as a sub-pipeline."
                )

    def _check_size_warning(
        self, pipeline: Pipeline, result: ValidationResult
    ) -> None:
        if len(pipeline) > self.LARGE_PIPELINE_THRESHOLD:
            result.add_warning(
                f"Pipeline has {len(pipeline)} steps (threshold: "
                f"{self.LARGE_PIPELINE_THRESHOLD}). "
                "Evaluation may be slow; consider increasing the parsimony penalty."
            )
