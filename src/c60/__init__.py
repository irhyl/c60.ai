"""
c60 — Molecular Evolution AutoML Framework

Public surface for Phase 1 (core data structures).
"""

__version__ = "0.2.0"

from c60.core.types import (
    DataType,
    TabularData,
    RawTabular,
    CleanTabular,
    ScaledData,
    UnscaledData,
    Features,
    Predictions,
    ClassLabels,
    RegressionValues,
    Probabilities,
    is_compatible,
    type_name,
)
from c60.core.pipeline import PipelineStep, Pipeline
from c60.core.registry import (
    OperationRegistry,
    OperationSpec,
    IntRange,
    FloatRange,
    Categorical,
    Boolean,
    default_registry,
)

__all__ = [
    # types
    "DataType", "TabularData", "RawTabular", "CleanTabular",
    "ScaledData", "UnscaledData", "Features",
    "Predictions", "ClassLabels", "RegressionValues", "Probabilities",
    "is_compatible", "type_name",
    # pipeline
    "PipelineStep", "Pipeline",
    # registry
    "OperationRegistry", "OperationSpec",
    "IntRange", "FloatRange", "Categorical", "Boolean",
    "default_registry",
]
