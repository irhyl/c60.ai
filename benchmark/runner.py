"""
runner.py — BenchmarkRunner: nested cross-validation across systems and datasets.

Protocol
--------
For each (system, dataset) pair:
  For each outer_seed in seeds:
    StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=outer_seed)
    For each fold:
      train / test split
      Fit system on train
      Score system on test
  Record mean ± std across all folds × seeds

This gives n_folds × len(seeds) accuracy estimates per (system, dataset),
making the mean robust to train/test split variance.

Results are stored in a pandas DataFrame with columns:
  system | dataset | seed | fold | accuracy | fit_time_s

Failed evaluations (exceptions) are recorded with accuracy=NaN and a note
in the errors dict, so one crash does not abort the whole benchmark.
"""

from __future__ import annotations

import time
import traceback
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_datasets() -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Return a list of (name, X, y) classification datasets.

    Uses sklearn built-ins only — no network access required.

    Datasets selected to cover diverse regimes:
      iris           — low-dimensional, 3-class, trivial (sanity check)
      wine           — moderate features, 3-class, strong baseline possible
      breast_cancer  — many features, binary, feature selection matters
      digits         — high-dimensional (64), 10-class, dim reduction helps
    """
    from sklearn.datasets import (
        load_breast_cancer,
        load_digits,
        load_iris,
        load_wine,
    )

    iris = load_iris(return_X_y=True)
    wine = load_wine(return_X_y=True)
    bc   = load_breast_cancer(return_X_y=True)
    dig  = load_digits(return_X_y=True)

    return [
        ("iris",           iris[0],  iris[1]),
        ("wine",           wine[0],  wine[1]),
        ("breast_cancer",  bc[0],    bc[1]),
        ("digits",         dig[0],   dig[1]),
    ]


def load_extended_datasets(
    max_samples: int = 8000,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Return larger, more challenging classification datasets via OpenML.

    All datasets are capped at *max_samples* rows (stratified) so that
    individual fold evaluations stay tractable.

    Dataset profiles
    ----------------
    pendigits     10 992 x  16   10 classes   pen-based digit recognition
    letter        20 000 x  16   26 classes   letter image recognition
    waveform       5 000 x  40    3 classes   waveform classification (Breiman)
    mnist_784     70 000 x 784   10 classes   handwritten digits (capped 8 000)
    covtype      581 012 x  54    7 classes   forest cover type (capped 8 000)
    har            10 299 x 561    6 classes   Human Activity Recognition (UCI)

    Falls back to make_classification synthetic data if OpenML is unavailable.
    """
    from sklearn.datasets import fetch_openml, make_classification
    from sklearn.utils import resample

    def _fetch(name, version, target_col=None, max_n=max_samples):
        ds = fetch_openml(
            name, version=version, as_frame=False,
            parser="auto", cache=True,
        )
        X = ds.data.astype(np.float32)
        y_raw = ds.target

        # Encode string targets to integers
        if y_raw.dtype.kind in ("U", "S", "O"):
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y_raw).astype(np.int32)
        else:
            y = y_raw.astype(np.int32)

        # Stratified cap
        if len(X) > max_n:
            X, y = resample(
                X, y,
                n_samples=max_n,
                stratify=y,
                random_state=0,
            )
        return X, y

    datasets = []

    _specs = [
        # (openml_name,  version,  our_name)
        ("pendigits",    1,        "pendigits"),
        ("letter",       1,        "letter"),
        ("waveform-5000", 1,       "waveform"),
        ("mnist_784",    1,        "mnist"),
        ("covertype",    3,        "covtype"),
        ("har",          1,        "har"),
    ]

    for oml_name, ver, our_name in _specs:
        try:
            X, y = _fetch(oml_name, ver)
            datasets.append((our_name, X, y))
            print(f"  Loaded {our_name:12s}: {X.shape[0]:6d} x {X.shape[1]:4d}, "
                  f"{len(np.unique(y))} classes")
        except Exception as exc:
            print(f"  Skipping {our_name}: {exc}")

    if not datasets:
        print("  OpenML unavailable — falling back to synthetic datasets.")
        for i, (n_f, n_c, name) in enumerate([
            (20, 5,  "synth_20f_5c"),
            (50, 10, "synth_50f_10c"),
            (100, 3, "synth_100f_3c"),
        ]):
            X, y = make_classification(
                n_samples=3000, n_features=n_f, n_informative=n_f//2,
                n_classes=n_c, n_clusters_per_class=1,
                random_state=i,
            )
            datasets.append((name, X.astype(np.float32), y.astype(np.int32)))

    return datasets


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Run a comparative benchmark with nested cross-validation.

    Parameters
    ----------
    n_folds : int
        Number of outer CV folds (default 5).
    seeds : list of int
        Random seeds for the outer fold shuffles.  Using multiple seeds
        reduces variance in the final estimates (default [0, 1, 2, 3, 4]).
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        n_folds: int = 5,
        seeds: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> None:
        self.n_folds = n_folds
        self.seeds = seeds if seeds is not None else list(range(5))
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        systems: List[Tuple[str, BaseEstimator]],
        datasets: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None,
    ) -> pd.DataFrame:
        """
        Run the full benchmark.

        Parameters
        ----------
        systems  : list of (name, estimator) — all systems to evaluate
        datasets : list of (name, X, y)  — defaults to load_datasets()

        Returns
        -------
        pd.DataFrame with columns:
            system, dataset, seed, fold, accuracy, fit_time_s
        """
        if datasets is None:
            datasets = load_datasets()

        records = []
        errors: Dict[str, str] = {}

        n_total = len(systems) * len(datasets) * len(self.seeds) * self.n_folds
        done = 0

        for ds_name, X, y in datasets:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Dataset: {ds_name}  "
                      f"({X.shape[0]} samples × {X.shape[1]} features, "
                      f"{len(np.unique(y))} classes)")
                print("="*60)

            for sys_name, estimator in systems:
                sys_accs = []

                for seed in self.seeds:
                    skf = StratifiedKFold(
                        n_splits=self.n_folds,
                        shuffle=True,
                        random_state=seed,
                    )

                    for fold_idx, (train_idx, test_idx) in enumerate(
                        skf.split(X, y)
                    ):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        acc, fit_time = self._evaluate_one(
                            estimator, sys_name, ds_name, seed, fold_idx,
                            X_train, y_train, X_test, y_test, errors,
                        )

                        records.append({
                            "system":     sys_name,
                            "dataset":    ds_name,
                            "seed":       seed,
                            "fold":       fold_idx,
                            "accuracy":   acc,
                            "fit_time_s": fit_time,
                        })

                        if acc is not None:
                            sys_accs.append(acc)

                        done += 1
                        if self.verbose:
                            pct = 100 * done / n_total
                            acc_str = f"{acc:.4f}" if acc is not None else "FAIL"
                            print(
                                f"  [{pct:5.1f}%] {sys_name:<22} "
                                f"ds={ds_name:<14} seed={seed} fold={fold_idx} "
                                f"acc={acc_str}  ({fit_time:.1f}s)"
                            )

                if sys_accs and self.verbose:
                    print(
                        f"  >> {sys_name:<22} mean={np.mean(sys_accs):.4f} "
                        f"+/- {np.std(sys_accs):.4f}"
                    )

        df = pd.DataFrame(records)
        if errors and self.verbose:
            print(f"\n{len(errors)} evaluation(s) failed:")
            for key, msg in errors.items():
                print(f"  {key}: {msg[:120]}")

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_one(
        self,
        estimator: BaseEstimator,
        sys_name: str,
        ds_name: str,
        seed: int,
        fold_idx: int,
        X_train, y_train,
        X_test, y_test,
        errors: dict,
    ) -> Tuple[Optional[float], float]:
        """Fit and score one (system, fold) pair. Returns (accuracy, fit_time)."""
        # Deep-clone so each fold gets a fresh, unfitted estimator
        try:
            est = clone(estimator)
        except Exception:
            est = deepcopy(estimator)

        # Propagate seed to C60Estimator so different folds still get
        # different (but reproducible) random behaviour
        if hasattr(est, "random_seed"):
            est.random_seed = seed * 100 + fold_idx

        t0 = time.perf_counter()
        try:
            est.fit(X_train, y_train)
            acc = float(est.score(X_test, y_test))
            fit_time = time.perf_counter() - t0
            return acc, fit_time
        except Exception:  # noqa: BLE001 — intentional: record any failure as NaN
            fit_time = time.perf_counter() - t0
            key = f"{sys_name}|{ds_name}|seed={seed}|fold={fold_idx}"
            errors[key] = traceback.format_exc(limit=3)
            return None, fit_time
