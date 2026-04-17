"""
hybrid — Neuro-symbolic pipeline nodes for C60.ai.

Exposes PyTorch-backed ML components that implement the sklearn
fit/transform/predict interface, making them first-class citizens
in any C60.ai Pipeline:

  NeuralAutoencoder   — unsupervised bottleneck encoder (dim_reducer)
  NeuralClassifier    — supervised MLP classifier

Registration
------------
Call register_hybrid_nodes() to add these to the default registry
so the GA can evolve pipelines that include neural nodes:

    from c60.hybrid.registry_ext import register_hybrid_nodes
    register_hybrid_nodes()   # extends default_registry in-place
"""

from c60.hybrid.node import NeuralAutoencoder, NeuralClassifier
from c60.hybrid.registry_ext import register_hybrid_nodes

__all__ = [
    "NeuralAutoencoder",
    "NeuralClassifier",
    "register_hybrid_nodes",
]
