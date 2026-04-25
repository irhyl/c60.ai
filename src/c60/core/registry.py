"""
registry.py — Operation Registry for C60.ai

The OperationRegistry is the central catalog of every ML operation that the
genetic operators can use when building or mutating pipeline nodes.

Each entry is an OperationSpec describing:
  - The sklearn (or compatible) class to instantiate
  - Its step_type label (must match the StepType taxonomy)
  - The DataType it consumes and produces
  - Its hyperparameter search space

Genetic operators query the registry by step_type to find valid replacement
or insertion candidates, then sample a concrete operation + hyperparameters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline as SklearnPipeline  # noqa: F401 (for reference)
from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from c60.core.types import (
    CleanTabular,
    DataType,
    Features,
    ClassLabels,
    RawTabular,
    RegressionValues,
    ScaledData,
    UnscaledData,
)
from c60.core.pipeline import PipelineStep


# ---------------------------------------------------------------------------
# Hyperparameter space descriptors
# ---------------------------------------------------------------------------

@dataclass
class IntRange:
    """Uniform integer range [low, high] (inclusive)."""
    low: int
    high: int

    def sample(self) -> int:
        return random.randint(self.low, self.high)


@dataclass
class FloatRange:
    """Log-uniform float range [low, high]."""
    low: float
    high: float
    log_scale: bool = False

    def sample(self) -> float:
        if self.log_scale:
            import math
            log_low = math.log10(self.low)
            log_high = math.log10(self.high)
            return 10 ** random.uniform(log_low, log_high)
        return random.uniform(self.low, self.high)


@dataclass
class Categorical:
    """Uniform choice from a finite set."""
    choices: List[Any]

    def sample(self) -> Any:
        return random.choice(self.choices)


@dataclass
class Boolean:
    """Random True/False."""
    def sample(self) -> bool:
        return random.random() < 0.5


# ---------------------------------------------------------------------------
# OperationSpec
# ---------------------------------------------------------------------------

@dataclass
class OperationSpec:
    """
    Full specification for one ML operation that can appear in a pipeline node.

    Attributes
    ----------
    op_class : type
        The class to instantiate (e.g. RandomForestClassifier).
    step_type : str
        The StepType taxonomy label (e.g. "classifier").
    input_type : Type[DataType]
        Data type this operation accepts.
    output_type : Type[DataType]
        Data type this operation produces.
    search_space : dict
        Maps hyperparameter name → a space descriptor (IntRange, FloatRange,
        Categorical, Boolean).  Only params listed here are mutated by the
        HyperparameterMutation operator.
    default_params : dict
        Fixed constructor kwargs that are NOT mutated (e.g. n_jobs=-1).
    """
    op_class: type
    step_type: str
    input_type: Type[DataType]
    output_type: Type[DataType]
    search_space: Dict[str, Any] = field(default_factory=dict)
    default_params: Dict[str, Any] = field(default_factory=dict)

    def sample_params(self) -> Dict[str, Any]:
        """Sample one concrete hyperparameter configuration from the search space."""
        return {k: v.sample() for k, v in self.search_space.items()}

    def instantiate(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a concrete instance of op_class with the given params merged
        on top of default_params.  Samples randomly if params is None.
        """
        kwargs = {**self.default_params}
        kwargs.update(params if params is not None else self.sample_params())
        return self.op_class(**kwargs)


# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------

_SPECS: List[OperationSpec] = [

    # -----------------------------------------------------------------------
    # Imputers  (RawTabular → CleanTabular)
    # -----------------------------------------------------------------------
    OperationSpec(
        op_class=SimpleImputer,
        step_type="imputer",
        input_type=RawTabular,
        output_type=CleanTabular,
        search_space={"strategy": Categorical(["mean", "median", "most_frequent"])},
    ),

    # -----------------------------------------------------------------------
    # Scalers  (CleanTabular → ScaledData)
    # -----------------------------------------------------------------------
    OperationSpec(
        op_class=StandardScaler,
        step_type="scaler",
        input_type=CleanTabular,
        output_type=ScaledData,
        search_space={"with_mean": Boolean(), "with_std": Boolean()},
    ),
    OperationSpec(
        op_class=MinMaxScaler,
        step_type="scaler",
        input_type=CleanTabular,
        output_type=ScaledData,
        search_space={},
    ),
    OperationSpec(
        op_class=RobustScaler,
        step_type="scaler",
        input_type=CleanTabular,
        output_type=ScaledData,
        search_space={
            "with_centering": Boolean(),
            "with_scaling": Boolean(),
        },
    ),

    # -----------------------------------------------------------------------
    # Feature engineering  (ScaledData → Features)
    # -----------------------------------------------------------------------
    OperationSpec(
        op_class=PolynomialFeatures,
        step_type="feature_engineer",
        input_type=ScaledData,
        output_type=Features,
        search_space={
            "degree": IntRange(2, 3),
            "interaction_only": Boolean(),
            "include_bias": Boolean(),
        },
    ),

    # -----------------------------------------------------------------------
    # Feature selection  (ScaledData → Features)
    # -----------------------------------------------------------------------
    OperationSpec(
        op_class=SelectKBest,
        step_type="feature_selector",
        input_type=ScaledData,
        output_type=Features,
        search_space={"k": IntRange(2, 20)},
        default_params={"score_func": f_classif},
    ),
    OperationSpec(
        op_class=SelectKBest,
        step_type="feature_selector_regression",
        input_type=ScaledData,
        output_type=Features,
        search_space={"k": IntRange(2, 20)},
        default_params={"score_func": f_regression},
    ),

    # -----------------------------------------------------------------------
    # Dimensionality reduction  (ScaledData → Features)
    # -----------------------------------------------------------------------
    OperationSpec(
        op_class=PCA,
        step_type="dim_reducer",
        input_type=ScaledData,
        output_type=Features,
        search_space={
            "n_components": FloatRange(0.7, 0.99),  # variance explained
            "whiten": Boolean(),
        },
    ),
    OperationSpec(
        op_class=TruncatedSVD,
        step_type="dim_reducer",
        input_type=ScaledData,
        output_type=Features,
        search_space={"n_components": IntRange(2, 50)},
    ),

    # -----------------------------------------------------------------------
    # Classifiers  (ScaledData | Features → ClassLabels)
    # -----------------------------------------------------------------------
    OperationSpec(
        op_class=LogisticRegression,
        step_type="classifier",
        input_type=ScaledData,
        output_type=ClassLabels,
        search_space={
            "C": FloatRange(1e-3, 1e3, log_scale=True),
            "penalty": Categorical(["l2", "l1"]),
            "solver": Categorical(["lbfgs", "liblinear", "saga"]),
            "max_iter": IntRange(100, 1000),
        },
        default_params={"random_state": 42},
    ),
    OperationSpec(
        op_class=RandomForestClassifier,
        step_type="classifier",
        input_type=UnscaledData,
        output_type=ClassLabels,
        search_space={
            "n_estimators": IntRange(50, 500),
            "max_depth": IntRange(2, 30),
            "min_samples_split": IntRange(2, 20),
            "criterion": Categorical(["gini", "entropy"]),
        },
        default_params={"random_state": 42, "n_jobs": -1},
    ),
    OperationSpec(
        op_class=GradientBoostingClassifier,
        step_type="classifier",
        input_type=UnscaledData,
        output_type=ClassLabels,
        search_space={
            "n_estimators": IntRange(50, 300),
            "max_depth": IntRange(2, 10),
            "learning_rate": FloatRange(1e-3, 0.5, log_scale=True),
            "subsample": FloatRange(0.5, 1.0),
        },
        default_params={"random_state": 42},
    ),
    OperationSpec(
        op_class=SVC,
        step_type="classifier",
        input_type=ScaledData,
        output_type=ClassLabels,
        search_space={
            "C": FloatRange(1e-2, 1e3, log_scale=True),
            "kernel": Categorical(["rbf", "linear", "poly"]),
            "gamma": Categorical(["scale", "auto"]),
        },
        default_params={"random_state": 42, "probability": True},
    ),
    OperationSpec(
        op_class=KNeighborsClassifier,
        step_type="classifier",
        input_type=ScaledData,
        output_type=ClassLabels,
        search_space={
            "n_neighbors": IntRange(1, 30),
            "weights": Categorical(["uniform", "distance"]),
            "p": Categorical([1, 2]),
        },
    ),
    OperationSpec(
        op_class=DecisionTreeClassifier,
        step_type="classifier",
        input_type=UnscaledData,
        output_type=ClassLabels,
        search_space={
            "max_depth": IntRange(2, 20),
            "criterion": Categorical(["gini", "entropy"]),
            "min_samples_split": IntRange(2, 20),
        },
        default_params={"random_state": 42},
    ),

    # -----------------------------------------------------------------------
    # Regressors  (ScaledData | Features → RegressionValues)
    # -----------------------------------------------------------------------
    OperationSpec(
        op_class=LinearRegression,
        step_type="regressor",
        input_type=ScaledData,
        output_type=RegressionValues,
        search_space={"fit_intercept": Boolean()},
    ),
    OperationSpec(
        op_class=Ridge,
        step_type="regressor",
        input_type=ScaledData,
        output_type=RegressionValues,
        search_space={
            "alpha": FloatRange(1e-3, 1e3, log_scale=True),
            "fit_intercept": Boolean(),
        },
    ),
    OperationSpec(
        op_class=RandomForestRegressor,
        step_type="regressor",
        input_type=UnscaledData,
        output_type=RegressionValues,
        search_space={
            "n_estimators": IntRange(50, 500),
            "max_depth": IntRange(2, 30),
            "min_samples_split": IntRange(2, 20),
        },
        default_params={"random_state": 42, "n_jobs": -1},
    ),
    OperationSpec(
        op_class=GradientBoostingRegressor,
        step_type="regressor",
        input_type=UnscaledData,
        output_type=RegressionValues,
        search_space={
            "n_estimators": IntRange(50, 300),
            "max_depth": IntRange(2, 10),
            "learning_rate": FloatRange(1e-3, 0.5, log_scale=True),
        },
        default_params={"random_state": 42},
    ),
    OperationSpec(
        op_class=SVR,
        step_type="regressor",
        input_type=ScaledData,
        output_type=RegressionValues,
        search_space={
            "C": FloatRange(1e-2, 1e3, log_scale=True),
            "kernel": Categorical(["rbf", "linear"]),
            "epsilon": FloatRange(1e-3, 1.0, log_scale=True),
        },
    ),
    OperationSpec(
        op_class=KNeighborsRegressor,
        step_type="regressor",
        input_type=ScaledData,
        output_type=RegressionValues,
        search_space={
            "n_neighbors": IntRange(1, 30),
            "weights": Categorical(["uniform", "distance"]),
        },
    ),
    OperationSpec(
        op_class=DecisionTreeRegressor,
        step_type="regressor",
        input_type=UnscaledData,
        output_type=RegressionValues,
        search_space={
            "max_depth": IntRange(2, 20),
            "min_samples_split": IntRange(2, 20),
            "criterion": Categorical(["squared_error", "absolute_error"]),
        },
        default_params={"random_state": 42},
    ),
]


# ---------------------------------------------------------------------------
# OperationRegistry
# ---------------------------------------------------------------------------

class OperationRegistry:
    """
    Central catalog of all available ML operations.

    Usage
    -----
    registry = OperationRegistry()

    # List all specs for a step type
    specs = registry.get_by_type("classifier")

    # Sample a random PipelineStep of a given type
    step = registry.sample_step("scaler")

    # Get the spec for a specific class
    spec = registry.get_spec(StandardScaler)
    """

    def __init__(self, specs: Optional[List[OperationSpec]] = None) -> None:
        self._specs: List[OperationSpec] = specs if specs is not None else list(_SPECS)

        # Build index: step_type → list of specs
        self._by_type: Dict[str, List[OperationSpec]] = {}
        for spec in self._specs:
            self._by_type.setdefault(spec.step_type, []).append(spec)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_by_type(self, step_type: str) -> List[OperationSpec]:
        """Return all specs registered for `step_type`."""
        return list(self._by_type.get(step_type, []))

    def get_spec(self, op_class: type) -> Optional[OperationSpec]:
        """Return the spec for a specific class, or None if not registered."""
        for spec in self._specs:
            if spec.op_class is op_class:
                return spec
        return None

    def all_step_types(self) -> List[str]:
        """Return the sorted list of registered step type labels."""
        return sorted(self._by_type.keys())

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_step(
        self,
        step_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> PipelineStep:
        """
        Sample a random PipelineStep of the given step_type.

        Picks a random OperationSpec for that type, samples hyperparameters
        (or uses provided `params`), and returns a ready-to-use PipelineStep.

        Raises
        ------
        ValueError
            If no specs are registered for `step_type`.
        """
        specs = self.get_by_type(step_type)
        if not specs:
            raise ValueError(
                f"No operations registered for step_type='{step_type}'. "
                f"Available types: {self.all_step_types()}"
            )
        spec = random.choice(specs)
        sampled_params = params if params is not None else spec.sample_params()
        operation = spec.instantiate(sampled_params)
        return PipelineStep(
            name=spec.op_class.__name__,
            step_type=spec.step_type,
            operation=operation,
            params=sampled_params,
            input_type=spec.input_type,
            output_type=spec.output_type,
        )

    def sample_compatible_step(
        self,
        input_type: Type[DataType],
        output_type: Optional[Type[DataType]] = None,
    ) -> Optional[PipelineStep]:
        """
        Sample a random step whose declared input_type is compatible with
        `input_type` (i.e. input_type IS-A step.input_type) and optionally
        whose output_type matches `output_type`.

        Returns None if no compatible spec is found.
        """
        from c60.core.types import is_compatible as compat

        candidates = [
            spec for spec in self._specs
            if compat(input_type, spec.input_type)
            and (output_type is None or compat(spec.output_type, output_type))
        ]
        if not candidates:
            return None
        spec = random.choice(candidates)
        sampled_params = spec.sample_params()
        operation = spec.instantiate(sampled_params)
        return PipelineStep(
            name=spec.op_class.__name__,
            step_type=spec.step_type,
            operation=operation,
            params=sampled_params,
            input_type=spec.input_type,
            output_type=spec.output_type,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: OperationSpec) -> None:
        """Add a new OperationSpec at runtime (e.g. custom user operations)."""
        self._specs.append(spec)
        self._by_type.setdefault(spec.step_type, []).append(spec)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["OperationRegistry"]
        for step_type in self.all_step_types():
            specs = self._by_type[step_type]
            names = ", ".join(s.op_class.__name__ for s in specs)
            lines.append(f"  {step_type:30s} ({len(specs):2d})  {names}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"OperationRegistry("
            f"{len(self._specs)} specs, "
            f"{len(self._by_type)} step types)"
        )


# ---------------------------------------------------------------------------
# Module-level default instance
# ---------------------------------------------------------------------------

#: The default registry — import and use directly, or subclass / extend.
default_registry = OperationRegistry()
