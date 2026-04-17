"""
node.py — Neuro-symbolic hybrid pipeline nodes.

Two concrete hybrid steps that implement the sklearn estimator protocol
so they plug directly into PipelineStep.operation:

  NeuralAutoencoder  — unsupervised 2-layer autoencoder used as a
                       non-linear dimensionality reducer / feature compressor.
                       fit() trains to minimise MSE reconstruction loss.
                       transform() returns the bottleneck (encoder output).
                       output_type: EmbeddedData (IS-A ScaledData → feeds classifiers)

  NeuralClassifier   — supervised 2-layer MLP used as a drop-in classifier.
                       fit() trains with cross-entropy.
                       predict() returns class labels.
                       predict_proba() returns softmax probabilities.
                       output_type: ClassLabels

Both classes:
  - Are CPU-only by default (avoids CUDA setup in tests and CI)
  - Raise ImportError with an install hint if torch is unavailable
  - Are deepcopy-safe (Pipeline.clone() uses copy.deepcopy)
  - Implement get_params() / set_params() for HyperparameterMutation
  - Set n_features_in_ after fit() for PipelineIntrospector
  - Are deterministic given random_state

The neuro-symbolic integration point
-------------------------------------
The typed DAG enforces that EmbeddedData IS-A ScaledData, so a
NeuralAutoencoder output can feed any downstream step that accepts
ScaledData — sklearn classifiers, SVMs, KNN, etc. — without any
special-casing. Neural and symbolic nodes are interchangeable at
compile time; the GA discovers which combination performs best.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# _require_torch — deferred import with helpful error
# ---------------------------------------------------------------------------

def _require_torch():
    """Import and return torch; raise ImportError if not installed."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "NeuralAutoencoder and NeuralClassifier require PyTorch.  "
            "Install it with: pip install torch"
        ) from exc


# ---------------------------------------------------------------------------
# Internal PyTorch architectures (defined lazily inside fit())
# ---------------------------------------------------------------------------

def _build_autoencoder_net(n_features: int, hidden_dim: int, bottleneck_dim: int):
    """Return an (encoder, decoder) tuple of nn.Sequential modules."""
    torch = _require_torch()
    import torch.nn as nn
    encoder = nn.Sequential(
        nn.Linear(n_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, bottleneck_dim),
        nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.Linear(bottleneck_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_features),
    )
    return encoder, decoder


def _build_mlp_net(n_features: int, hidden_dim: int, n_classes: int):
    """Return a single nn.Sequential MLP for classification."""
    torch = _require_torch()
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, n_classes),
    )


# ---------------------------------------------------------------------------
# NeuralAutoencoder
# ---------------------------------------------------------------------------

class NeuralAutoencoder:
    """
    Unsupervised neural feature compressor (sklearn-compatible transformer).

    Trains a 2-layer autoencoder on X and uses the encoder half as a
    non-linear feature extractor.  Serves as a drop-in replacement for
    PCA / TruncatedSVD inside a C60.ai Pipeline.

    Parameters
    ----------
    hidden_dim : int
        Width of the hidden layer in both encoder and decoder.
    bottleneck_dim : int
        Dimensionality of the compressed representation (encoder output).
    epochs : int
        Number of full passes over the training data.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size for SGD.
    random_state : int
        Seed for PyTorch's RNG — ensures reproducibility.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        bottleneck_dim: int = 8,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

        # Populated after fit()
        self._encoder = None
        self._decoder = None
        self.n_features_in_: Optional[int] = None
        self.reconstruction_loss_: Optional[float] = None

    # ------------------------------------------------------------------
    # sklearn transformer protocol
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y=None) -> "NeuralAutoencoder":
        torch = _require_torch()
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        X_arr = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        self._encoder, self._decoder = _build_autoencoder_net(
            n_features, self.hidden_dim, self.bottleneck_dim
        )

        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_arr)
        last_loss = float("inf")

        self._encoder.train()
        self._decoder.train()
        for _ in range(self.epochs):
            # Simple full-batch for small datasets; mini-batch for larger ones
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = X_t[batch_idx]
                opt.zero_grad()
                z = self._encoder(xb)
                x_hat = self._decoder(z)
                loss = criterion(x_hat, xb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            last_loss = epoch_loss / max(n_batches, 1)

        self._encoder.eval()
        self._decoder.eval()
        self.reconstruction_loss_ = last_loss
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError(
                "NeuralAutoencoder must be fitted before transform()."
            )
        torch = _require_torch()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            z = self._encoder(X_t)
        return z.numpy()

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # sklearn estimator protocol (get_params / set_params)
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "NeuralAutoencoder":
        for k, v in params.items():
            setattr(self, k, v)
        # Invalidate the fitted model when hyperparameters change
        self._encoder = None
        self._decoder = None
        self.n_features_in_ = None
        return self

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NeuralAutoencoder("
            f"hidden={self.hidden_dim}, "
            f"bottleneck={self.bottleneck_dim}, "
            f"epochs={self.epochs})"
        )

    def __deepcopy__(self, memo):
        """Custom deepcopy that safely copies PyTorch modules."""
        torch = _require_torch()
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ("_encoder", "_decoder") and v is not None:
                import io
                buf = io.BytesIO()
                torch.save(v.state_dict(), buf)
                buf.seek(0)
                # Rebuild the same architecture and load weights
                if k == "_encoder":
                    enc, _ = _build_autoencoder_net(
                        self.n_features_in_, self.hidden_dim, self.bottleneck_dim
                    )
                    enc.load_state_dict(torch.load(buf, weights_only=True))
                    enc.eval()
                    setattr(new, k, enc)
                else:
                    _, dec = _build_autoencoder_net(
                        self.n_features_in_, self.hidden_dim, self.bottleneck_dim
                    )
                    dec.load_state_dict(torch.load(buf, weights_only=True))
                    dec.eval()
                    setattr(new, k, dec)
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new


# ---------------------------------------------------------------------------
# NeuralClassifier
# ---------------------------------------------------------------------------

class NeuralClassifier:
    """
    Supervised MLP classifier (sklearn-compatible).

    A 2-layer fully-connected network trained with cross-entropy loss.
    Serves as a drop-in replacement for LogisticRegression / SVC inside
    a C60.ai Pipeline, participating in the same type system, mutation
    operators, and fitness evaluation as any sklearn classifier.

    Parameters
    ----------
    hidden_dim : int
        Width of the single hidden layer.
    epochs : int
        Number of training epochs.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

        # Populated after fit()
        self._net = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_classes_: Optional[int] = None

    # ------------------------------------------------------------------
    # sklearn classifier protocol
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralClassifier":
        torch = _require_torch()
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self.n_classes_ = n_classes

        # Encode labels to contiguous 0 … n_classes-1
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([label_to_idx[c] for c in y_arr], dtype=np.int64)

        self._net = _build_mlp_net(n_features, self.hidden_dim, n_classes)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_arr)
        y_t = torch.tensor(y_enc)

        self._net.train()
        for _ in range(self.epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb, yb = X_t[batch_idx], y_t[batch_idx]
                opt.zero_grad()
                logits = self._net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

        self._net.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError(
                "NeuralClassifier must be fitted before predict()."
            )
        torch = _require_torch()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            logits = self._net(X_t)
            proba = torch.softmax(logits, dim=1)
        return proba.numpy()

    # ------------------------------------------------------------------
    # sklearn estimator protocol
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "NeuralClassifier":
        for k, v in params.items():
            setattr(self, k, v)
        self._net = None
        self.classes_ = None
        self.n_features_in_ = None
        return self

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NeuralClassifier("
            f"hidden={self.hidden_dim}, "
            f"epochs={self.epochs}, "
            f"lr={self.lr})"
        )

    def __deepcopy__(self, memo):
        """Custom deepcopy that safely copies the PyTorch network."""
        torch = _require_torch()
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "_net" and v is not None:
                import io
                buf = io.BytesIO()
                torch.save(v.state_dict(), buf)
                buf.seek(0)
                net_copy = _build_mlp_net(
                    self.n_features_in_, self.hidden_dim, self.n_classes_
                )
                net_copy.load_state_dict(torch.load(buf, weights_only=True))
                net_copy.eval()
                setattr(new, k, net_copy)
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new
