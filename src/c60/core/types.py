"""
types.py — DataType lattice and compatibility system for C60.ai

Every PipelineStep declares what data type it consumes and what it produces.
An edge (A → B) is only valid if A's output_type is a subtype of B's input_type.
This prevents nonsensical pipelines (e.g. Classifier → Scaler) from ever
being constructed, keeping the genetic operators in a valid search space.

Type Lattice (read top-down as "is a subtype of"):

    DataType
    ├── TabularData
    │   ├── RawTabular          raw input — may have NaN, categoricals, mixed dtypes
    │   └── CleanTabular        imputed + encoded, fully numeric
    │       ├── ScaledData      normalized / standardized CleanTabular
    │       └── UnscaledData    CleanTabular that has NOT been scaled
    ├── Features                output of feature engineering / selection / dim-reduction
    ├── Predictions
    │   ├── ClassLabels         discrete class predictions
    │   └── RegressionValues    continuous predictions
    └── Probabilities           predict_proba output (n_samples × n_classes)
"""

from __future__ import annotations
from typing import Type

# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------

class DataType:
    """Root of the data-type lattice. No step declares this directly."""

# ---------------------------------------------------------------------------
# Tabular branch
# ---------------------------------------------------------------------------

class TabularData(DataType):
    """Any 2-D numeric-compatible array (n_samples × n_features)."""

class RawTabular(TabularData):
    """
    Raw input data as it comes from the user.
    May contain NaN values, categorical columns, or mixed dtypes.
    Only Imputer and Encoder steps accept this type.
    """

class CleanTabular(TabularData):
    """
    Fully imputed, encoded, and numeric-only tabular data.
    Ready for scaling, feature engineering, or direct model input.
    """

class ScaledData(CleanTabular):
    """
    CleanTabular data that has been normalized or standardized.
    Preferred input for distance-based models (SVM, KNN) and linear models.
    """

class UnscaledData(CleanTabular):
    """
    CleanTabular data that has NOT been scaled.
    Accepted by tree-based models that do not require scaling.
    """

# ---------------------------------------------------------------------------
# Feature branch
# ---------------------------------------------------------------------------

class Features(DataType):
    """
    Output of a feature engineering, selection, or dimensionality reduction step.
    Conceptually a 2-D numeric array, but semantically distinct from raw tabular.
    """

class EmbeddedData(ScaledData):
    """
    Output of a neural feature extractor or autoencoder.
    Represents a learned non-linear embedding of the input.
    IS-A ScaledData: compatible with any step that accepts scaled or clean tabular data,
    enabling seamless mixing of neural and symbolic pipeline nodes.
    """


# ---------------------------------------------------------------------------
# Prediction branch
# ---------------------------------------------------------------------------

class Predictions(DataType):
    """Base type for any terminal model output."""


class ClassLabels(Predictions):
    """Discrete class predictions (1-D integer or string array)."""


class RegressionValues(Predictions):
    """Continuous scalar predictions (1-D float array)."""


# ---------------------------------------------------------------------------
# Probability branch
# ---------------------------------------------------------------------------

class Probabilities(DataType):
    """
    Class probability estimates from predict_proba (n_samples × n_classes).
    Used as input to Ensemblers that operate on soft votes.
    """


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def is_compatible(output_type: Type[DataType], input_type: Type[DataType]) -> bool:
    """
    Return True if `output_type` is a subtype of `input_type` in the lattice.

    An edge (A → B) is valid iff:
        is_compatible(A.output_type, B.input_type) == True

    Examples:
        is_compatible(ScaledData, CleanTabular)  → True   (ScaledData IS-A CleanTabular)
        is_compatible(CleanTabular, ScaledData)  → False  (CleanTabular is NOT ScaledData)
        is_compatible(ClassLabels, Predictions)  → True
        is_compatible(ClassLabels, Features)     → False
    """
    return issubclass(output_type, input_type)


def compatible_input_types(output_type: Type[DataType]) -> list[Type[DataType]]:
    """
    Return all types in the lattice that `output_type` is compatible with
    (i.e., all ancestor types including itself).  Useful for registry lookups.
    """
    lattice = [
        DataType,
        TabularData, RawTabular, CleanTabular, ScaledData, UnscaledData, EmbeddedData,
        Features,
        Predictions, ClassLabels, RegressionValues,
        Probabilities,
    ]
    return [t for t in lattice if issubclass(output_type, t)]


# ---------------------------------------------------------------------------
# Human-readable names (used in logging and PipelineStory)
# ---------------------------------------------------------------------------

TYPE_NAMES: dict[Type[DataType], str] = {
    DataType:          "Data",
    TabularData:       "Tabular",
    RawTabular:        "RawTabular",
    CleanTabular:      "CleanTabular",
    ScaledData:        "ScaledData",
    UnscaledData:      "UnscaledData",
    EmbeddedData:      "EmbeddedData",
    Features:          "Features",
    Predictions:       "Predictions",
    ClassLabels:       "ClassLabels",
    RegressionValues:  "RegressionValues",
    Probabilities:     "Probabilities",
}


def type_name(t: Type[DataType]) -> str:
    return TYPE_NAMES.get(t, t.__name__)
