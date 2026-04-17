"""
registry_ext.py — Extend the default OperationRegistry with hybrid neural nodes.

Call register_hybrid_nodes() once at application startup (or in a notebook)
to make NeuralAutoencoder and NeuralClassifier available to the GA:

    from c60.hybrid.registry_ext import register_hybrid_nodes
    register_hybrid_nodes()   # mutates default_registry in-place

After this call, the EvolutionEngine can evolve pipelines that include
neural nodes alongside sklearn nodes.  The type system ensures neural
outputs (EmbeddedData) are only connected to compatible downstream steps.

If torch is not installed, register_hybrid_nodes() is a no-op (it logs a
warning but does not raise, so code that conditionally registers is safe).
"""

from __future__ import annotations

from typing import Optional

from c60.core.registry import (
    Categorical,
    FloatRange,
    IntRange,
    OperationRegistry,
    OperationSpec,
    default_registry,
)
from c60.core.types import ClassLabels, EmbeddedData, ScaledData


def register_hybrid_nodes(registry: Optional[OperationRegistry] = None) -> bool:
    """
    Register NeuralAutoencoder and NeuralClassifier in the given registry.

    Parameters
    ----------
    registry : OperationRegistry, optional
        Target registry.  Defaults to default_registry.

    Returns
    -------
    bool
        True if registration succeeded, False if torch is unavailable.
    """
    try:
        import torch as _  # noqa: F401 — verify torch is importable
    except ImportError:
        import warnings
        warnings.warn(
            "register_hybrid_nodes(): torch is not installed — "
            "NeuralAutoencoder and NeuralClassifier will not be registered.  "
            "Install PyTorch with: pip install torch",
            stacklevel=2,
        )
        return False

    from c60.hybrid.node import NeuralAutoencoder, NeuralClassifier

    reg = registry if registry is not None else default_registry

    # -----------------------------------------------------------------------
    # NeuralAutoencoder  (ScaledData → EmbeddedData)
    # -----------------------------------------------------------------------
    # Registered as a "dim_reducer" so the GA can swap it with PCA/TruncatedSVD.
    # EmbeddedData IS-A ScaledData so its output connects to any downstream
    # classifier or transformer that accepts ScaledData.
    reg.register(OperationSpec(
        op_class=NeuralAutoencoder,
        step_type="dim_reducer",
        input_type=ScaledData,
        output_type=EmbeddedData,
        search_space={
            "hidden_dim":     Categorical([16, 32, 64]),
            "bottleneck_dim": Categorical([4, 8, 16]),
            "epochs":         IntRange(5, 30),
            "lr":             FloatRange(1e-4, 1e-2, log_scale=True),
        },
        default_params={
            "hidden_dim": 32, "bottleneck_dim": 8,
            "epochs": 20, "lr": 1e-3,
            "batch_size": 32, "random_state": 42,
        },
    ))

    # -----------------------------------------------------------------------
    # NeuralClassifier  (ScaledData → ClassLabels)
    # -----------------------------------------------------------------------
    # Registered as a "classifier" so the GA can swap it with LR/SVM/RF.
    # Also accepts EmbeddedData input because EmbeddedData IS-A ScaledData.
    reg.register(OperationSpec(
        op_class=NeuralClassifier,
        step_type="classifier",
        input_type=ScaledData,
        output_type=ClassLabels,
        search_space={
            "hidden_dim": Categorical([32, 64, 128]),
            "epochs":     IntRange(10, 50),
            "lr":         FloatRange(1e-4, 1e-2, log_scale=True),
        },
        default_params={
            "hidden_dim": 64, "epochs": 30, "lr": 1e-3,
            "batch_size": 32, "random_state": 42,
        },
    ))

    return True
