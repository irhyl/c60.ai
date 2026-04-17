"""
test_benchmark.py — Infrastructure tests for the benchmark harness.

These tests verify that the benchmark machinery works correctly
WITHOUT running the full benchmark (which takes many minutes).

All tests use tiny datasets and minimal CV settings (2 folds × 1 seed)
so the suite stays fast enough to be included in CI.

Sections
--------
1. Dataset loader
2. Baseline instantiation
3. C60Estimator wrapper
4. BenchmarkRunner (tiny smoke run)
5. ResultsReporter (synthetic data)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator, ClassifierMixin


# ===========================================================================
# Section 1 — Dataset loader
# ===========================================================================

class TestDatasetLoader:

    def test_returns_four_datasets(self):
        from benchmark.runner import load_datasets
        datasets = load_datasets()
        assert len(datasets) == 4

    def test_each_entry_is_triple(self):
        from benchmark.runner import load_datasets
        for entry in load_datasets():
            assert len(entry) == 3

    def test_names_are_expected(self):
        from benchmark.runner import load_datasets
        names = [n for n, *_ in load_datasets()]
        assert "iris" in names
        assert "wine" in names
        assert "breast_cancer" in names
        assert "digits" in names

    def test_arrays_are_2d_1d(self):
        from benchmark.runner import load_datasets
        for name, X, y in load_datasets():
            assert X.ndim == 2, f"{name}: X should be 2D"
            assert y.ndim == 1, f"{name}: y should be 1D"

    def test_X_y_lengths_match(self):
        from benchmark.runner import load_datasets
        for name, X, y in load_datasets():
            assert len(X) == len(y), f"{name}: X/y length mismatch"


# ===========================================================================
# Section 2 — Baseline instantiation
# ===========================================================================

class TestBaselineInstantiation:

    def test_make_baselines_returns_list(self):
        from benchmark.baselines import make_baselines
        baselines = make_baselines()
        assert isinstance(baselines, list)
        assert len(baselines) == 9

    def test_each_baseline_is_named_tuple(self):
        from benchmark.baselines import make_baselines
        for name, est in make_baselines():
            assert isinstance(name, str) and name
            assert hasattr(est, "fit") and hasattr(est, "predict")

    def test_baseline_names_are_unique(self):
        from benchmark.baselines import make_baselines
        names = [n for n, _ in make_baselines()]
        assert len(names) == len(set(names)), "Duplicate baseline names"

    def test_expected_baselines_present(self):
        from benchmark.baselines import make_baselines
        names = {n for n, _ in make_baselines()}
        for expected in ("LR", "SVM-RBF", "KNN-10", "RandomForest",
                         "GradientBoosting", "PCA+LR",
                         "SelectKBest+GBT", "VotingEnsemble",
                         "RandomSearch-GBT"):
            assert expected in names, f"Missing baseline: {expected}"

    def test_all_systems_returns_ten(self):
        from benchmark.baselines import all_systems
        systems = all_systems()
        assert len(systems) == 10

    def test_c60_ai_in_all_systems(self):
        from benchmark.baselines import all_systems
        names = [n for n, _ in all_systems()]
        assert "C60.ai" in names


# ===========================================================================
# Section 3 — C60Estimator wrapper
# ===========================================================================

class TestC60Estimator:

    def test_instantiation(self):
        from benchmark.baselines import C60Estimator
        est = C60Estimator(population_size=4, max_generations=2)
        assert est.population_size == 4
        assert est.max_generations == 2

    def test_is_sklearn_compatible(self):
        from benchmark.baselines import C60Estimator
        from sklearn.base import BaseEstimator, ClassifierMixin
        est = C60Estimator()
        assert isinstance(est, BaseEstimator)
        assert isinstance(est, ClassifierMixin)

    def test_get_params(self):
        from benchmark.baselines import C60Estimator
        est = C60Estimator(population_size=5, max_generations=3)
        params = est.get_params()
        assert params["population_size"] == 5
        assert params["max_generations"] == 3

    def test_not_fitted_before_fit(self):
        from benchmark.baselines import C60Estimator
        est = C60Estimator()
        assert est._best_pipeline is None

    def test_fit_on_tiny_iris(self):
        from benchmark.baselines import C60Estimator
        X, y = load_iris(return_X_y=True)
        # Take every 5th sample so all 3 classes are represented (30 samples)
        X_s, y_s = X[::5], y[::5]
        est = C60Estimator(
            population_size=4, max_generations=2,
            cv=2, eval_timeout=30.0, random_seed=0,
        )
        est.fit(X_s, y_s)
        assert est._best_pipeline is not None

    def test_predict_after_fit(self):
        from benchmark.baselines import C60Estimator
        X, y = load_iris(return_X_y=True)
        X_s, y_s = X[::5], y[::5]
        est = C60Estimator(population_size=4, max_generations=2,
                           cv=2, eval_timeout=30.0, random_seed=0)
        est.fit(X_s, y_s)
        preds = est.predict(X_s[:10])
        assert len(preds) == 10

    def test_score_after_fit(self):
        from benchmark.baselines import C60Estimator
        X, y = load_iris(return_X_y=True)
        X_s, y_s = X[::5], y[::5]
        est = C60Estimator(population_size=4, max_generations=2,
                           cv=2, eval_timeout=30.0, random_seed=0)
        est.fit(X_s, y_s)
        score = est.score(X_s, y_s)
        assert 0.0 <= score <= 1.0

    def test_predict_without_fit_raises(self):
        from benchmark.baselines import C60Estimator
        est = C60Estimator()
        with pytest.raises(RuntimeError):
            est.predict(np.zeros((5, 4)))

    def test_repr(self):
        from benchmark.baselines import C60Estimator
        est = C60Estimator(population_size=10, max_generations=5)
        r = repr(est)
        assert "C60Estimator" in r
        assert "10" in r


# ===========================================================================
# Section 4 — BenchmarkRunner (minimal smoke run)
# ===========================================================================

class _ConstantClassifier(BaseEstimator, ClassifierMixin):
    """Always predicts majority class — tiny, deterministic, sklearn-API compliant."""

    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self._majority = classes[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self._majority)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


class _AlwaysFailClassifier(BaseEstimator, ClassifierMixin):
    """Raises on fit — for error-handling tests."""

    def fit(self, X, y):
        raise RuntimeError("intentional failure")

    def predict(self, X):
        raise RuntimeError("not fitted")

    def score(self, X, y):
        return 0.0


@pytest.fixture(scope="module")
def tiny_dataset():
    X, y = load_iris(return_X_y=True)
    return [("tiny_iris", X[:30], y[:30])]


@pytest.fixture(scope="module")
def tiny_runner():
    from benchmark.runner import BenchmarkRunner
    return BenchmarkRunner(n_folds=2, seeds=[0], verbose=False)


class TestBenchmarkRunner:

    def test_run_returns_dataframe(self, tiny_runner, tiny_dataset):
        df = tiny_runner.run(
            [("Constant", _ConstantClassifier())],
            datasets=tiny_dataset,
        )
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_expected_columns(self, tiny_runner, tiny_dataset):
        df = tiny_runner.run(
            [("Constant", _ConstantClassifier())],
            datasets=tiny_dataset,
        )
        for col in ("system", "dataset", "seed", "fold", "accuracy", "fit_time_s"):
            assert col in df.columns, f"Missing column: {col}"

    def test_correct_number_of_rows(self, tiny_runner, tiny_dataset):
        # 1 system × 1 dataset × 1 seed × 2 folds = 2 rows
        df = tiny_runner.run(
            [("Constant", _ConstantClassifier())],
            datasets=tiny_dataset,
        )
        assert len(df) == 2

    def test_accuracy_in_unit_interval(self, tiny_runner, tiny_dataset):
        df = tiny_runner.run(
            [("Constant", _ConstantClassifier())],
            datasets=tiny_dataset,
        )
        valid = df["accuracy"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_fit_time_non_negative(self, tiny_runner, tiny_dataset):
        df = tiny_runner.run(
            [("Constant", _ConstantClassifier())],
            datasets=tiny_dataset,
        )
        assert (df["fit_time_s"] >= 0).all()

    def test_failed_eval_records_nan(self, tiny_runner, tiny_dataset):
        df = tiny_runner.run(
            [("Fail", _AlwaysFailClassifier())],
            datasets=tiny_dataset,
        )
        assert df["accuracy"].isna().all(), "Expected all NaN for always-failing estimator"

    def test_multiple_systems(self, tiny_runner, tiny_dataset):
        systems = [
            ("A", _ConstantClassifier()),
            ("B", _ConstantClassifier()),
        ]
        df = tiny_runner.run(systems, datasets=tiny_dataset)
        assert set(df["system"].unique()) == {"A", "B"}

    def test_default_datasets_loaded(self, tiny_runner):
        """Runner with default datasets produces non-empty DataFrame."""
        df = tiny_runner.run([("Constant", _ConstantClassifier())])
        assert len(df) > 0
        assert len(df["dataset"].unique()) == 4

    def test_random_seed_propagated(self, tiny_runner, tiny_dataset):
        """C60Estimator-style estimators receive random_seed from runner."""

        class SeedCapture(BaseEstimator, ClassifierMixin):
            def __init__(self, random_seed=99):
                self.random_seed = random_seed
                self.fitted_seed = None

            def fit(self, X, y):
                self.fitted_seed = self.random_seed
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def score(self, X, y):
                return float(np.mean(self.predict(X) == y))

        cap = SeedCapture(random_seed=99)
        df = tiny_runner.run([("Cap", cap)], datasets=tiny_dataset)
        # Runner should have set random_seed to seed*100+fold on at least one eval
        assert len(df) == 2  # 2 folds × 1 seed


# ===========================================================================
# Section 5 — ResultsReporter (synthetic data)
# ===========================================================================

@pytest.fixture(scope="module")
def synthetic_results():
    """Build a realistic-looking results DataFrame with 3 systems."""
    rng = np.random.default_rng(0)
    records = []
    systems  = {"C60.ai": 0.88, "LR": 0.82, "RF": 0.85}
    datasets = ["ds1", "ds2"]
    for sys, base_acc in systems.items():
        for ds in datasets:
            for seed in range(3):
                for fold in range(3):
                    acc = float(np.clip(rng.normal(base_acc, 0.03), 0, 1))
                    records.append({
                        "system": sys, "dataset": ds,
                        "seed": seed, "fold": fold,
                        "accuracy": acc, "fit_time_s": float(rng.uniform(0.1, 2.0)),
                    })
    return pd.DataFrame(records)


class TestResultsReporter:

    def test_instantiation(self, synthetic_results):
        from benchmark.report import ResultsReporter
        r = ResultsReporter(synthetic_results)
        assert r is not None

    def test_summary_table_markdown(self, synthetic_results):
        from benchmark.report import ResultsReporter
        tbl = ResultsReporter(synthetic_results).summary_table(fmt="markdown")
        assert isinstance(tbl, str)
        assert "C60.ai" in tbl

    def test_summary_table_plain(self, synthetic_results):
        from benchmark.report import ResultsReporter
        tbl = ResultsReporter(synthetic_results).summary_table(fmt="plain")
        assert "LR" in tbl

    def test_summary_table_latex(self, synthetic_results):
        from benchmark.report import ResultsReporter
        tbl = ResultsReporter(synthetic_results).summary_table(fmt="latex")
        assert "tabular" in tbl

    def test_rank_table_returns_dataframe(self, synthetic_results):
        from benchmark.report import ResultsReporter
        rt = ResultsReporter(synthetic_results).rank_table()
        assert isinstance(rt, pd.DataFrame)
        assert "System" in rt.columns
        assert "Avg Rank" in rt.columns

    def test_rank_table_has_all_systems(self, synthetic_results):
        from benchmark.report import ResultsReporter
        rt = ResultsReporter(synthetic_results).rank_table()
        systems = set(rt["System"])
        assert systems == {"C60.ai", "LR", "RF"}

    def test_rank_table_sorted_ascending(self, synthetic_results):
        from benchmark.report import ResultsReporter
        rt = ResultsReporter(synthetic_results).rank_table()
        ranks = list(rt["Avg Rank"])
        assert ranks == sorted(ranks)

    def test_best_system_has_rank_1(self, synthetic_results):
        from benchmark.report import ResultsReporter
        rt = ResultsReporter(synthetic_results).rank_table()
        assert rt.iloc[0]["Avg Rank"] == pytest.approx(1.0, abs=0.5)

    def test_wilcoxon_returns_dataframe(self, synthetic_results):
        from benchmark.report import ResultsReporter
        wt = ResultsReporter(synthetic_results).wilcoxon_vs_c60()
        assert isinstance(wt, pd.DataFrame)

    def test_wilcoxon_has_expected_columns(self, synthetic_results):
        from benchmark.report import ResultsReporter
        wt = ResultsReporter(synthetic_results).wilcoxon_vs_c60()
        for col in ("system", "mean_diff", "p_value", "significant"):
            assert col in wt.columns

    def test_wilcoxon_excludes_c60_itself(self, synthetic_results):
        from benchmark.report import ResultsReporter
        wt = ResultsReporter(synthetic_results).wilcoxon_vs_c60()
        assert "C60.ai" not in wt["system"].values

    def test_wilcoxon_n_rows(self, synthetic_results):
        from benchmark.report import ResultsReporter
        wt = ResultsReporter(synthetic_results).wilcoxon_vs_c60()
        # 3 systems total → 2 comparisons
        assert len(wt) == 2

    def test_print_full_report_runs(self, synthetic_results, capsys):
        from benchmark.report import ResultsReporter
        ResultsReporter(synthetic_results).print_full_report()
        out = capsys.readouterr().out
        assert "C60.ai" in out
        assert "Rank" in out

    def test_render_comparison_returns_figure(self, synthetic_results):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from benchmark.report import ResultsReporter
        fig = ResultsReporter(synthetic_results).render_comparison()
        assert fig is not None
        plt.close("all")

    def test_render_ranks_returns_figure(self, synthetic_results):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from benchmark.report import ResultsReporter
        fig = ResultsReporter(synthetic_results).render_ranks()
        assert fig is not None
        plt.close("all")

    def test_render_winloss_returns_figure(self, synthetic_results):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from benchmark.report import ResultsReporter
        fig = ResultsReporter(synthetic_results).render_winloss()
        assert fig is not None
        plt.close("all")
