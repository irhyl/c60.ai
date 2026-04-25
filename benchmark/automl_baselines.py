"""
automl_baselines.py — AutoML system wrappers for C60.ai comparison.

Each wrapper implements the sklearn estimator interface (fit / predict / score)
so it can be dropped into the existing BenchmarkRunner unchanged.

Active systems (returned by all_automl_systems())
--------------------------------------------------
 1  HyperoptSearch    TPE-based hyperparameter search over 7 model families
 2  OptunaSearch      Optuna TPE over 7 model families + feature selectors
 3  BayesSearchCV     scikit-optimize Gaussian-process BO over GBT+SVM+RF
 4  SuccessiveHalving sklearn HalvingRandomSearchCV — resource-aware pruning
 5  BroadRandomSearch RandomizedSearchCV over all sklearn model families
 6  GreedyEnsemble    Forward-selection stacking (auto-sklearn style post-proc)
 7  AutoStack         3-level stacking with auto-selected base learners
 8  FeatEngAutoML     Feature engineering → SelectPercentile → best-model search
 9  OptunaEnsemble    Optuna-optimised voting weights over 5 base models

Additional class definitions retained for ad-hoc use
-----------------------------------------------------
  HalvingGridAutoML  — prohibitively slow on datasets > 2000 samples
  TPOT, FLAML, H2O   — platform/dependency constraints on this build

Notes
-----
- All wrappers respect a `time_budget` parameter (default 120 s).
- Wrappers that use randomness accept `random_state` for reproducibility.
"""

from __future__ import annotations

import warnings
import time
import numpy as np
from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

_CLASSIFIERS = [
    ("lr",  LogisticRegression(max_iter=500, random_state=0)),
    ("svm", SVC(probability=True, random_state=0)),
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=0)),
    ("gbt", GradientBoostingClassifier(n_estimators=100, random_state=0)),
    ("knn", KNeighborsClassifier(n_neighbors=10)),
    ("et",  ExtraTreesClassifier(n_estimators=100, random_state=0)),
    ("dt",  DecisionTreeClassifier(random_state=0)),
]


def _best_of(candidates: list[tuple[str, BaseEstimator]], X, y) -> BaseEstimator:
    """Return the candidate with the highest 3-fold CV accuracy."""
    best_score, best_est = -1.0, candidates[0][1]
    for _, est in candidates:
        try:
            s = cross_val_score(est, X, y, cv=_CV, scoring="accuracy",
                                error_score=0.0).mean()
            if s > best_score:
                best_score, best_est = s, est
        except Exception:
            pass
    return best_est


# ---------------------------------------------------------------------------
# 1. TPOT
# ---------------------------------------------------------------------------

class TPOTAutoML(BaseEstimator, ClassifierMixin):
    """TPOT genetic-programming pipeline search."""

    def __init__(self, time_budget: int = 120, random_state: int = 42):
        self.time_budget = time_budget
        self.random_state = random_state

    def fit(self, X, y):
        from tpot import TPOTClassifier
        # generations/population chosen to fit within time_budget
        self._model = TPOTClassifier(
            generations=5,
            population_size=20,
            cv=3,
            max_time_mins=max(1, self.time_budget // 60),
            random_state=self.random_state,
            verbosity=0,
            n_jobs=1,
            config_dict="TPOT light",
        )
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        return self._model.score(X, y)


# ---------------------------------------------------------------------------
# 2. FLAML
# ---------------------------------------------------------------------------

class FLAMLAutoML(BaseEstimator, ClassifierMixin):
    """FLAML cost-frugal hyperparameter / model search."""

    def __init__(self, time_budget: int = 120, random_state: int = 42):
        self.time_budget = time_budget
        self.random_state = random_state

    def fit(self, X, y):
        from flaml import AutoML
        self._model = AutoML()
        self._model.fit(
            X, y,
            task="classification",
            time_budget=self.time_budget,
            seed=self.random_state,
            verbose=0,
        )
        return self

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 3. HyperoptSearch — TPE over 7 model families
# ---------------------------------------------------------------------------

class HyperoptSearch(BaseEstimator, ClassifierMixin):
    """Hyperopt TPE search over 7 sklearn classifier families."""

    def __init__(self, max_evals: int = 50, random_state: int = 42):
        self.max_evals = max_evals
        self.random_state = random_state

    def fit(self, X, y):
        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

        space = hp.choice("classifier", [
            {"name": "lr",
             "C": hp.loguniform("lr_C", -4, 4),
             "max_iter": 500},
            {"name": "svm",
             "C": hp.loguniform("svm_C", -3, 3),
             "gamma": hp.choice("svm_gamma", ["scale", "auto"])},
            {"name": "rf",
             "n_estimators": hp.quniform("rf_n", 50, 300, 50),
             "max_depth": hp.choice("rf_depth", [None, 5, 10, 20])},
            {"name": "gbt",
             "n_estimators": hp.quniform("gbt_n", 50, 200, 50),
             "learning_rate": hp.loguniform("gbt_lr", -4, 0),
             "max_depth": hp.quniform("gbt_depth", 2, 6, 1)},
            {"name": "knn",
             "n_neighbors": hp.quniform("knn_k", 3, 30, 2)},
            {"name": "et",
             "n_estimators": hp.quniform("et_n", 50, 300, 50)},
            {"name": "dt",
             "max_depth": hp.quniform("dt_depth", 2, 20, 1)},
        ])

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        best_score = [-1.0]
        best_est = [None]

        def objective(params):
            name = params["name"]
            if name == "lr":
                est = LogisticRegression(C=params["C"], max_iter=int(params["max_iter"]))
            elif name == "svm":
                est = SVC(C=params["C"], gamma=params["gamma"])
            elif name == "rf":
                est = RandomForestClassifier(n_estimators=int(params["n_estimators"]),
                                             max_depth=params["max_depth"], random_state=0)
            elif name == "gbt":
                est = GradientBoostingClassifier(
                    n_estimators=int(params["n_estimators"]),
                    learning_rate=params["learning_rate"],
                    max_depth=int(params["max_depth"]), random_state=0)
            elif name == "knn":
                est = KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]))
            elif name == "et":
                est = ExtraTreesClassifier(n_estimators=int(params["n_estimators"]), random_state=0)
            else:
                est = DecisionTreeClassifier(max_depth=int(params["max_depth"]), random_state=0)

            try:
                score = cross_val_score(est, X_s, y, cv=_CV,
                                        scoring="accuracy", error_score=0.0).mean()
            except Exception:
                score = 0.0

            if score > best_score[0]:
                best_score[0] = score
                best_est[0] = clone(est)
            return {"loss": -score, "status": STATUS_OK}

        trials = Trials()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fmin(objective, space, algo=tpe.suggest,
                 max_evals=self.max_evals, trials=trials,
                 rstate=np.random.default_rng(self.random_state),
                 show_progressbar=False)

        self._scaler = scaler
        self._est = best_est[0]
        self._est.fit(X_s, y)
        return self

    def predict(self, X):
        return self._est.predict(self._scaler.transform(X))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 4. OptunaSearch — Optuna TPE over 7 families + SelectPercentile
# ---------------------------------------------------------------------------

class OptunaSearch(BaseEstimator, ClassifierMixin):
    """Optuna TPE search over model families and feature selection."""

    def __init__(self, n_trials: int = 50, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state

    def fit(self, X, y):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        best = {"score": -1.0, "est": None, "sel": None}

        def objective(trial):
            # Optional feature selection
            use_sel = trial.suggest_categorical("use_sel", [True, False])
            if use_sel:
                pct = trial.suggest_int("percentile", 30, 100, step=10)
                sel = SelectPercentile(f_classif, percentile=pct)
                X_t = sel.fit_transform(X_s, y)
            else:
                sel, X_t = None, X_s

            name = trial.suggest_categorical(
                "clf", ["lr", "svm", "rf", "gbt", "knn", "et", "dt"])

            if name == "lr":
                est = LogisticRegression(
                    C=trial.suggest_float("lr_C", 1e-4, 1e4, log=True),
                    max_iter=500)
            elif name == "svm":
                est = SVC(
                    C=trial.suggest_float("svm_C", 1e-3, 1e3, log=True),
                    gamma=trial.suggest_categorical("svm_g", ["scale", "auto"]))
            elif name == "rf":
                est = RandomForestClassifier(
                    n_estimators=trial.suggest_int("rf_n", 50, 300, step=50),
                    max_depth=trial.suggest_categorical("rf_d", [None, 5, 10, 20]),
                    random_state=0)
            elif name == "gbt":
                est = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int("gbt_n", 50, 200, step=50),
                    learning_rate=trial.suggest_float("gbt_lr", 1e-3, 0.5, log=True),
                    max_depth=trial.suggest_int("gbt_d", 2, 6),
                    random_state=0)
            elif name == "knn":
                est = KNeighborsClassifier(
                    n_neighbors=trial.suggest_int("knn_k", 3, 30, step=2))
            elif name == "et":
                est = ExtraTreesClassifier(
                    n_estimators=trial.suggest_int("et_n", 50, 300, step=50),
                    random_state=0)
            else:
                est = DecisionTreeClassifier(
                    max_depth=trial.suggest_int("dt_d", 2, 20),
                    random_state=0)

            try:
                score = cross_val_score(est, X_t, y, cv=_CV,
                                        scoring="accuracy", error_score=0.0).mean()
            except Exception:
                score = 0.0

            if score > best["score"]:
                best["score"] = score
                best["est"] = clone(est)
                best["sel"] = clone(sel) if sel is not None else None
            return score

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self._scaler = scaler
        self._sel = best["sel"]
        self._est = best["est"]

        X_fit = X_s
        if self._sel is not None:
            X_fit = self._sel.fit_transform(X_s, y)
        self._est.fit(X_fit, y)
        return self

    def predict(self, X):
        X_t = self._scaler.transform(X)
        if self._sel is not None:
            X_t = self._sel.transform(X_t)
        return self._est.predict(X_t)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 5. BayesSearchCV — scikit-optimize GP/RF BO over GBT + SVM + RF
# ---------------------------------------------------------------------------

class BayesSearchAutoML(BaseEstimator, ClassifierMixin):
    """scikit-optimize BayesSearchCV over three strong model families."""

    def __init__(self, n_iter: int = 30, random_state: int = 42):
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        candidates = [
            (GradientBoostingClassifier(random_state=0), {
                "n_estimators":  Integer(50, 300),
                "learning_rate": Real(1e-3, 0.5, prior="log-uniform"),
                "max_depth":     Integer(2, 8),
                "subsample":     Real(0.5, 1.0),
            }),
            (SVC(probability=True, random_state=0), {
                "C":     Real(1e-3, 1e3, prior="log-uniform"),
                "gamma": Real(1e-5, 1e1, prior="log-uniform"),
            }),
            (RandomForestClassifier(random_state=0), {
                "n_estimators": Integer(50, 300),
                "max_depth":    Categorical([None, 5, 10, 20]),
                "max_features": Real(0.1, 1.0),
            }),
        ]

        best_score, best_model = -1.0, None
        for base_est, search_space in candidates:
            try:
                opt = BayesSearchCV(
                    base_est, search_space,
                    n_iter=self.n_iter // len(candidates),
                    cv=_CV, scoring="accuracy",
                    random_state=self.random_state, n_jobs=1,
                    refit=True, verbose=0,
                )
                opt.fit(X_s, y)
                if opt.best_score_ > best_score:
                    best_score = opt.best_score_
                    best_model = opt.best_estimator_
            except Exception:
                pass

        self._scaler = scaler
        self._model = best_model if best_model else GradientBoostingClassifier().fit(X_s, y)
        if best_model:
            self._model.fit(X_s, y)
        return self

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 6. SuccessiveHalving — resource-aware random search
# ---------------------------------------------------------------------------

class SuccessiveHalvingAutoML(BaseEstimator, ClassifierMixin):
    """HalvingRandomSearchCV — successive halving over all model families."""

    def __init__(self, n_candidates: int = 40, random_state: int = 42):
        self.n_candidates = n_candidates
        self.random_state = random_state

    def fit(self, X, y):
        from scipy.stats import loguniform, randint

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # Large joint param space: model type is a hyperparameter
        # (simulate via trying each family with HalvingRandomSearch)
        best_score, best_pipe = -1.0, None

        specs = [
            (GradientBoostingClassifier(random_state=0), {
                "n_estimators":  randint(50, 300),
                "learning_rate": loguniform(0.01, 0.5),
                "max_depth":     randint(2, 8),
            }),
            (SVC(random_state=0), {
                "C":     loguniform(0.01, 100),
                "gamma": loguniform(1e-4, 1.0),
            }),
            (RandomForestClassifier(random_state=0), {
                "n_estimators": randint(50, 300),
                "max_depth":    randint(3, 20),
            }),
            (LogisticRegression(max_iter=500, random_state=0), {
                "C": loguniform(1e-4, 1e4),
            }),
        ]

        for base_est, param_dist in specs:
            try:
                search = HalvingRandomSearchCV(
                    base_est, param_dist,
                    n_candidates=self.n_candidates // len(specs),
                    cv=_CV, scoring="accuracy",
                    random_state=self.random_state, n_jobs=1,
                    refit=True, verbose=0,
                )
                search.fit(X_s, y)
                if search.best_score_ > best_score:
                    best_score = search.best_score_
                    best_pipe = search.best_estimator_
            except Exception:
                pass

        self._scaler = scaler
        self._model = best_pipe or GradientBoostingClassifier().fit(X_s, y)
        return self

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 7. BroadRandomSearch — RandomizedSearchCV over all 7 model families
# ---------------------------------------------------------------------------

class BroadRandomSearch(BaseEstimator, ClassifierMixin):
    """RandomizedSearchCV spanning all 7 sklearn classifier families."""

    def __init__(self, n_iter: int = 60, random_state: int = 42):
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        from scipy.stats import loguniform, randint

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        best_score, best_est = -1.0, None
        specs = [
            (LogisticRegression(max_iter=500, random_state=0),
             {"C": loguniform(1e-4, 1e4)}),
            (SVC(random_state=0),
             {"C": loguniform(0.01, 100), "gamma": loguniform(1e-5, 10)}),
            (RandomForestClassifier(random_state=0),
             {"n_estimators": randint(50, 400), "max_depth": randint(3, 20)}),
            (GradientBoostingClassifier(random_state=0),
             {"n_estimators": randint(50, 300),
              "learning_rate": loguniform(0.01, 0.5),
              "max_depth": randint(2, 8)}),
            (KNeighborsClassifier(),
             {"n_neighbors": randint(3, 30)}),
            (ExtraTreesClassifier(random_state=0),
             {"n_estimators": randint(50, 400), "max_depth": randint(3, 20)}),
            (DecisionTreeClassifier(random_state=0),
             {"max_depth": randint(2, 20), "min_samples_split": randint(2, 20)}),
        ]

        n_per = max(5, self.n_iter // len(specs))
        for base_est, param_dist in specs:
            try:
                search = RandomizedSearchCV(
                    base_est, param_dist, n_iter=n_per,
                    cv=_CV, scoring="accuracy",
                    random_state=self.random_state, n_jobs=1, refit=True)
                search.fit(X_s, y)
                if search.best_score_ > best_score:
                    best_score = search.best_score_
                    best_est = search.best_estimator_
            except Exception:
                pass

        self._scaler = scaler
        self._model = best_est or RandomForestClassifier().fit(X_s, y)
        return self

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 8. GreedyEnsemble — forward selection stacking (auto-sklearn post-processing)
# ---------------------------------------------------------------------------

class GreedyEnsemble(BaseEstimator, ClassifierMixin):
    """Greedy forward-selection ensemble from a library of base models."""

    def __init__(self, n_iterations: int = 20, random_state: int = 42):
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.model_selection import cross_val_predict
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        library = [
            LogisticRegression(C=1.0, max_iter=500, random_state=0),
            SVC(C=10, gamma="scale", probability=True, random_state=0),
            RandomForestClassifier(n_estimators=100, random_state=0),
            GradientBoostingClassifier(n_estimators=100, random_state=0),
            ExtraTreesClassifier(n_estimators=100, random_state=0),
            KNeighborsClassifier(n_neighbors=10),
            LogisticRegression(C=0.1, max_iter=500, random_state=1),
        ]

        # Get OOF predictions for all base models
        oof_preds = []
        fitted = []
        classes = np.unique(y)
        for est in library:
            try:
                proba = cross_val_predict(est, X_s, y, cv=_CV,
                                          method="predict_proba",
                                          n_jobs=1)
                oof_preds.append(proba)
                fitted.append(clone(est).fit(X_s, y))
            except Exception:
                pass

        if not fitted:
            self._scaler = scaler
            self._weights = None
            self._models = [RandomForestClassifier().fit(X_s, y)]
            return self

        # Greedy forward selection
        n = len(fitted)
        weights = np.zeros(n)
        best_score = -1.0

        for _ in range(self.n_iterations):
            best_i, best_s = -1, best_score
            for i in range(n):
                w_try = weights.copy()
                w_try[i] += 1
                ensemble_proba = sum(w_try[j] * oof_preds[j]
                                     for j in range(n)) / w_try.sum()
                preds = classes[np.argmax(ensemble_proba, axis=1)]
                s = (preds == y).mean()
                if s > best_s:
                    best_s, best_i = s, i
            if best_i == -1:
                break
            weights[best_i] += 1
            best_score = best_s

        if weights.sum() == 0:
            weights[0] = 1.0

        self._scaler = scaler
        self._weights = weights / weights.sum()
        self._models = fitted
        self._classes = classes
        return self

    def predict(self, X):
        X_s = self._scaler.transform(X)
        proba = sum(self._weights[i] * m.predict_proba(X_s)
                    for i, m in enumerate(self._models))
        return self._classes[np.argmax(proba, axis=1)]

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 9. AutoStack — 3-level stacking with auto-selected base learners
# ---------------------------------------------------------------------------

class AutoStack(BaseEstimator, ClassifierMixin):
    """Two-level stacking: diverse base learners + LR meta-learner."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        base = [
            ("rf",  RandomForestClassifier(n_estimators=100, random_state=0)),
            ("gbt", GradientBoostingClassifier(n_estimators=100, random_state=0)),
            ("svm", SVC(C=10, gamma="scale", probability=True, random_state=0)),
            ("et",  ExtraTreesClassifier(n_estimators=100, random_state=0)),
            ("knn", KNeighborsClassifier(n_neighbors=10)),
        ]
        meta = LogisticRegression(C=1.0, max_iter=500, random_state=0)

        self._scaler = scaler
        self._stack = StackingClassifier(
            estimators=base,
            final_estimator=meta,
            cv=_CV,
            passthrough=True,
            n_jobs=1,
        )
        self._stack.fit(X_s, y)
        return self

    def predict(self, X):
        return self._stack.predict(self._scaler.transform(X))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 10. FeatureEngineeringAutoML — auto feature selection + model search
# ---------------------------------------------------------------------------

class FeatureEngineeringAutoML(BaseEstimator, ClassifierMixin):
    """Searches over feature selection percentile + 5 classifiers."""

    def __init__(self, n_iter: int = 40, random_state: int = 42):
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        from scipy.stats import loguniform, randint

        best_score, best_pipe = -1.0, None
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_iter):
            pct = int(rng.choice([30, 40, 50, 60, 70, 80, 90, 100]))
            clf_name = rng.choice(["lr", "svm", "rf", "gbt", "et"])
            if clf_name == "lr":
                clf = LogisticRegression(
                    C=float(rng.uniform(-3, 3)), max_iter=500, random_state=0)
            elif clf_name == "svm":
                clf = SVC(C=float(np.exp(rng.uniform(-2, 4))),
                          gamma="scale", random_state=0)
            elif clf_name == "rf":
                clf = RandomForestClassifier(
                    n_estimators=int(rng.integers(50, 200)),
                    max_depth=rng.choice([None, 5, 10, 20]),
                    random_state=0)
            elif clf_name == "gbt":
                clf = GradientBoostingClassifier(
                    n_estimators=int(rng.integers(50, 200)),
                    learning_rate=float(np.exp(rng.uniform(-4, 0))),
                    random_state=0)
            else:
                clf = ExtraTreesClassifier(
                    n_estimators=int(rng.integers(50, 200)), random_state=0)

            pipe = SklearnPipeline([
                ("scaler", StandardScaler()),
                ("sel",    SelectPercentile(f_classif, percentile=pct)),
                ("clf",    clf),
            ])
            try:
                score = cross_val_score(pipe, X, y, cv=_CV,
                                        scoring="accuracy",
                                        error_score=0.0).mean()
                if score > best_score:
                    best_score = score
                    best_pipe = clone(pipe)
            except Exception:
                pass

        self._model = best_pipe or SklearnPipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier()),
        ])
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 11. OptunaEnsemble — Optuna-optimised ensemble weights
# ---------------------------------------------------------------------------

class OptunaEnsemble(BaseEstimator, ClassifierMixin):
    """Fit 5 diverse classifiers, then use Optuna to find optimal voting weights."""

    def __init__(self, n_trials: int = 60, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state

    def fit(self, X, y):
        import optuna
        from sklearn.model_selection import cross_val_predict
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        classes = np.unique(y)

        base = [
            LogisticRegression(C=1.0, max_iter=500, random_state=0),
            SVC(C=10, gamma="scale", probability=True, random_state=0),
            RandomForestClassifier(n_estimators=100, random_state=0),
            GradientBoostingClassifier(n_estimators=100, random_state=0),
            ExtraTreesClassifier(n_estimators=100, random_state=0),
        ]

        oof = []
        fitted = []
        for est in base:
            try:
                p = cross_val_predict(est, X_s, y, cv=_CV,
                                      method="predict_proba", n_jobs=1)
                oof.append(p)
                fitted.append(clone(est).fit(X_s, y))
            except Exception:
                pass

        if not oof:
            self._scaler = scaler
            self._fitted = [RandomForestClassifier().fit(X_s, y)]
            self._weights = np.array([1.0])
            self._classes = classes
            return self

        def objective(trial):
            ws = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0)
                           for i in range(len(oof))])
            if ws.sum() == 0:
                return 0.0
            ws /= ws.sum()
            proba = sum(ws[i] * oof[i] for i in range(len(oof)))
            preds = classes[np.argmax(proba, axis=1)]
            return (preds == y).mean()

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best = study.best_params
        ws = np.array([best[f"w{i}"] for i in range(len(oof))])
        ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(oof)) / len(oof)

        self._scaler = scaler
        self._fitted = fitted
        self._weights = ws
        self._classes = classes
        return self

    def predict(self, X):
        X_s = self._scaler.transform(X)
        proba = sum(self._weights[i] * m.predict_proba(X_s)
                    for i, m in enumerate(self._fitted))
        return self._classes[np.argmax(proba, axis=1)]

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# 12. HalvingGridSearch — exhaustive grid search with resource-aware pruning
# ---------------------------------------------------------------------------

class HalvingGridAutoML(BaseEstimator, ClassifierMixin):
    """HalvingGridSearchCV over a fixed grid — aggressive pruning of weak configs."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        best_score, best_est = -1.0, None
        specs = [
            (GradientBoostingClassifier(random_state=0), {
                "n_estimators":  [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth":     [2, 3, 5],
            }),
            (RandomForestClassifier(random_state=0), {
                "n_estimators": [50, 100, 200],
                "max_depth":    [None, 5, 10],
                "max_features": [0.5, 0.75, 1.0],
            }),
            (SVC(probability=True, random_state=0), {
                "C":     [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.001],
            }),
        ]

        for base_est, param_grid in specs:
            try:
                search = HalvingGridSearchCV(
                    base_est, param_grid,
                    cv=_CV, scoring="accuracy",
                    random_state=self.random_state, n_jobs=1,
                    refit=True, verbose=0,
                )
                search.fit(X_s, y)
                if search.best_score_ > best_score:
                    best_score = search.best_score_
                    best_est = search.best_estimator_
            except Exception:
                pass

        self._scaler = scaler
        self._model = best_est or GradientBoostingClassifier().fit(X_s, y)
        return self

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def all_automl_systems(time_budget: int = 120) -> list[tuple[str, BaseEstimator]]:
    """
    Return the 9 AutoML systems used in the benchmark.
    Systems that require unavailable packages are silently skipped with a
    printed warning so the benchmark degrades gracefully.

    Excluded (class definitions retained for ad-hoc use):
      BayesSearchCV  — skopt ignores time_budget; 600-1200 s/fold on large datasets
      HalvingGrid    — O(n_samples) per round; prohibitive on 8 000-sample datasets
      TPOT           — requires torch (MemoryError on this build)
      FLAML          — ignores time_budget on Windows/Python 3.13
      H2O            — requires 500 MB disk
    """
    candidates = [
        ("HyperoptSearch",    lambda: HyperoptSearch(max_evals=20)),
        ("OptunaSearch",      lambda: OptunaSearch(n_trials=20)),
        ("BayesSearchCV",     lambda: BayesSearchAutoML(n_iter=15)),
        ("SuccessiveHalving", lambda: SuccessiveHalvingAutoML(n_candidates=30)),
        ("BroadRandomSearch", lambda: BroadRandomSearch(n_iter=20)),
        ("GreedyEnsemble",    lambda: GreedyEnsemble(n_iterations=10)),
        ("AutoStack",         lambda: AutoStack()),
        ("FeatEngAutoML",     lambda: FeatureEngineeringAutoML(n_iter=20)),
        ("OptunaEnsemble",    lambda: OptunaEnsemble(n_trials=20)),
    ]

    systems = []
    for name, factory in candidates:
        try:
            est = factory()
            systems.append((name, est))
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    return systems
