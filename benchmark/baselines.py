"""
baselines.py — Baseline pipeline definitions and the C60Estimator wrapper.

Ten systems are compared:

 #   Name                    Description
 ── ─────────────────────── ────────────────────────────────────────────────
 1  LR                       StandardScaler → LogisticRegression (C=1)
 2  SVM-RBF                  StandardScaler → SVC(kernel=rbf, C=10)
 3  KNN-10                   StandardScaler → KNeighborsClassifier(k=10)
 4  RandomForest             RandomForestClassifier(n=200, max_depth=None)
 5  GradientBoosting         GradientBoostingClassifier(n=200, lr=0.1)
 6  PCA+LR                   StandardScaler → PCA(0.95) → LogisticRegression
 7  SelectKBest+GBT          StandardScaler → SelectKBest(f_classif,k=10)
                             → GradientBoostingClassifier
 8  VotingEnsemble           Hard vote of LR + RF + GBT
 9  RandomSearch-GBT         RandomizedSearchCV over GBT (50 iters, 3-fold)
10  C60.ai                   EvolutionEngine (pop=15, gen=8, cv=3)

The first nine are fixed sklearn pipelines with manually chosen but reasonable
hyperparameters — they represent competent human baselines and common AutoML
starting points.  RandomSearch-GBT represents the simplest form of AutoML
(random hyperparameter search).  C60.ai represents full structural + parametric
search via genetic algorithms.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform, randint


# ---------------------------------------------------------------------------
# Utility: SelectKBest that clips k to the actual number of features
# ---------------------------------------------------------------------------

class _ClippedSelectKBest(SelectKBest):
    """SelectKBest that silently clips k to min(k, n_features) at fit time.

    Prevents ValueError on low-dimensional datasets (e.g. iris with 4 features
    when k=10 was requested).
    """

    def fit(self, X, y=None):  # type: ignore[override]
        n_features = X.shape[1]
        if self.k != "all" and self.k > n_features:
            self.k = n_features
        return super().fit(X, y)


# ---------------------------------------------------------------------------
# 1–9: Sklearn baselines
# ---------------------------------------------------------------------------

def make_baselines() -> List[Tuple[str, BaseEstimator]]:
    """
    Return a list of (name, estimator) pairs for all baseline systems.
    Each estimator implements fit / predict / score.
    """
    n_est      = 200
    n_est_vote = 100
    rs_iter    = 10

    # 1 — Logistic Regression
    lr = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])

    # 2 — SVM with RBF kernel
    svm = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                    random_state=42)),
    ])

    # 3 — K-Nearest Neighbours
    knn = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=10, weights="distance")),
    ])

    # 4 — Random Forest (no scaler — trees don't need it)
    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=None, min_samples_split=2,
        random_state=42, n_jobs=1,
    )

    # 5 — Gradient Boosting
    gbt = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=0.1, max_depth=3,
        subsample=0.8, random_state=42,
    )

    # 6 — PCA (95% variance) + Logistic Regression
    pca_lr = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=0.95, random_state=42)),
        ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])

    # 7 — Univariate feature selection (k=10) + Gradient Boosting
    skb_gbt = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("sel",    SelectKBest(score_func=f_classif, k=10)),
        ("clf",    GradientBoostingClassifier(
            n_estimators=n_est, learning_rate=0.1, max_depth=3,
            subsample=0.8, random_state=42,
        )),
    ])

    # 8 — Voting Ensemble (LR + RF + GBT, hard vote)
    voting = VotingClassifier(
        estimators=[
            ("lr",  LogisticRegression(C=1.0, max_iter=500, random_state=42)),
            ("rf",  RandomForestClassifier(n_estimators=n_est_vote, random_state=42,
                                           n_jobs=1)),
            ("gbt", GradientBoostingClassifier(
                n_estimators=n_est_vote, learning_rate=0.1, max_depth=3,
                random_state=42,
            )),
        ],
        voting="hard",
    )
    # Voting needs scaled input — wrap in a pipeline
    voting_pipe = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("clf",    voting),
    ])

    # 9 — RandomizedSearchCV over GradientBoosting (50 iterations, 3-fold)
    #     Represents naive single-model AutoML (hyperparameter search only)
    _param_dist = {
        "n_estimators":      randint(50, 400),
        "learning_rate":     loguniform(0.01, 0.3),
        "max_depth":         randint(2, 8),
        "subsample":         loguniform(0.5, 1.0),
        "min_samples_split": randint(2, 20),
    }
    rand_gbt_pipe = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomizedSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_distributions=_param_dist,
            n_iter=rs_iter,
            cv=3,
            scoring="accuracy",
            random_state=42,
            n_jobs=1,   # multiprocessing unavailable on some Windows builds
            refit=True,
        )),
    ])

    return [
        ("LR",                lr),
        ("SVM-RBF",           svm),
        ("KNN-10",            knn),
        ("RandomForest",      rf),
        ("GradientBoosting",  gbt),
        ("PCA+LR",            pca_lr),
        ("SelectKBest+GBT",   skb_gbt),
        ("VotingEnsemble",    voting_pipe),
        ("RandomSearch-GBT",  rand_gbt_pipe),
    ]


# ---------------------------------------------------------------------------
# 10 — C60.ai estimator (sklearn-compatible wrapper)
# ---------------------------------------------------------------------------

class C60Estimator(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around C60.ai EvolutionEngine.

    Wraps the full genetic algorithm pipeline search in the standard
    fit / predict / score interface so it can be used in cross_val_score
    alongside sklearn baselines.

    Parameters
    ----------
    population_size : int
        GA population size per generation.
    max_generations : int
        Maximum number of generations.
    cv : int
        Inner cross-validation folds used by FitnessEvaluator.
    eval_timeout : float
        Per-pipeline evaluation timeout in seconds.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 15,
        max_generations: int = 8,
        cv: int = 3,
        eval_timeout: float = 30.0,
        random_seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.max_generations = max_generations
        self.cv = cv
        self.eval_timeout = eval_timeout
        self.random_seed = random_seed
        self._best_pipeline = None
        self._log = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "C60Estimator":
        from c60.evolution.engine import EvolutionEngine
        engine = EvolutionEngine(
            population_size=self.population_size,
            max_generations=self.max_generations,
            cv=self.cv,
            task="classification",
            eval_timeout=self.eval_timeout,
            complexity_penalty=0.002,
            random_seed=self.random_seed,
        )
        best = engine.fit(X, y)
        # Refit the winning pipeline on the full training data so predict()
        # works after the benchmark runner calls fit() on each fold.
        best.fit(X, y)
        self._best_pipeline = best
        self._log = engine.history()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._best_pipeline is None:
            raise RuntimeError("C60Estimator must be fitted before predict().")
        return self._best_pipeline.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:  # type: ignore[override]
        return float(np.mean(self.predict(X) == y))

    @property
    def evolution_log(self):
        return self._log

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"C60Estimator(pop={self.population_size}, "
            f"gen={self.max_generations})"
        )


def all_systems() -> List[Tuple[str, BaseEstimator]]:
    """Return all 10 (name, estimator) pairs including C60.ai."""
    baselines = make_baselines()
    c60 = C60Estimator(
        population_size=15,
        max_generations=8,
        cv=3,
        eval_timeout=30.0,
        random_seed=42,
    )
    return baselines + [("C60.ai", c60)]
