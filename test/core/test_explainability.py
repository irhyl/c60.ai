"""
test_explainability.py — Tests for PipelineIntrospector, PipelineStory,
                          and the render_* visualisation helpers.

Structure
---------
Section 1 — StepReport
Section 2 — PipelineReport
Section 3 — PipelineIntrospector  (unfitted and fitted pipelines)
Section 4 — PipelineStory          (narrate, pipeline_summary, generation_table)
Section 5 — render_pipeline / render_evolution  (smoke tests)
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from c60.core.pipeline import Pipeline, PipelineStep
from c60.core.types import (
    CleanTabular, ScaledData, ClassLabels, RegressionValues, Features,
)
from c60.evaluation.cache import FitnessResult
from c60.evolution.engine import EvolutionLog, GenerationRecord
from c60.explainability.introspector import (
    PipelineIntrospector, PipelineReport, StepReport,
)
from c60.explainability.story import PipelineStory


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================

@pytest.fixture
def iris_data():
    return load_iris(return_X_y=True, as_frame=False)


@pytest.fixture
def iris_feature_names():
    return list(load_iris().feature_names)


@pytest.fixture
def simple_clf_pipeline():
    """Scaler → LogisticRegression (unfitted)."""
    p = Pipeline(name="SimpleCLF")
    sc = PipelineStep(
        name="StandardScaler", step_type="scaler",
        operation=StandardScaler(),
        input_type=CleanTabular, output_type=ScaledData,
    )
    clf = PipelineStep(
        name="LogisticRegression", step_type="classifier",
        operation=LogisticRegression(max_iter=300),
        input_type=ScaledData, output_type=ClassLabels,
    )
    p.add_step(sc)
    p.add_step(clf)
    p.add_edge(sc.id, clf.id)
    return p


@pytest.fixture
def rf_clf_pipeline():
    """Scaler → RandomForestClassifier (unfitted)."""
    p = Pipeline(name="RFCLF")
    sc = PipelineStep(
        name="StandardScaler", step_type="scaler",
        operation=StandardScaler(),
        input_type=CleanTabular, output_type=ScaledData,
    )
    rf = PipelineStep(
        name="RandomForestClassifier", step_type="classifier",
        operation=RandomForestClassifier(n_estimators=20, random_state=0),
        input_type=ScaledData, output_type=ClassLabels,
    )
    p.add_step(sc)
    p.add_step(rf)
    p.add_edge(sc.id, rf.id)
    return p


@pytest.fixture
def fitted_rf_pipeline(rf_clf_pipeline, iris_data):
    X, y = iris_data
    rf_clf_pipeline.fit(X, y)
    return rf_clf_pipeline


@pytest.fixture
def fitted_lr_pipeline(simple_clf_pipeline, iris_data):
    X, y = iris_data
    simple_clf_pipeline.fit(X, y)
    return simple_clf_pipeline


@pytest.fixture
def dummy_fitness():
    return FitnessResult(
        raw_score=0.95, adjusted_score=0.94,
        cv_scores=[0.93, 0.95, 0.96], wall_time=0.5, n_nodes=2,
    )


def _make_log(best_scores, mean_scores=None) -> EvolutionLog:
    """Build a fake EvolutionLog from score lists."""
    if mean_scores is None:
        mean_scores = [s * 0.9 for s in best_scores]
    log = EvolutionLog()
    for i, (b, m) in enumerate(zip(best_scores, mean_scores)):
        log.append(GenerationRecord(
            generation=i,
            best_score=b,
            mean_score=m,
            population_size=5,
            wall_time=float(i + 1) * 1.2,
            best_pipeline_hash=f"hash_{i}",
        ))
    return log


# ===========================================================================
# Section 1 — StepReport
# ===========================================================================

class TestStepReport:

    def _make_step_report(self, importances=None, source=None):
        return StepReport(
            step_id="abc",
            step_name="MyStep",
            step_type="classifier",
            operation_class="LogisticRegression",
            params={"C": 1.0},
            feature_importances=importances,
            importance_source=source,
        )

    def test_has_importances_false_when_none(self):
        sr = self._make_step_report()
        assert not sr.has_importances()

    def test_has_importances_true_when_set(self):
        sr = self._make_step_report(importances=np.array([0.3, 0.7]))
        assert sr.has_importances()

    def test_top_k_indices_order(self):
        imp = np.array([0.1, 0.5, 0.3, 0.05, 0.05])
        sr = self._make_step_report(importances=imp)
        indices = sr.top_k_indices(k=2)
        assert indices[0] == 1   # 0.5 is max
        assert indices[1] == 2   # 0.3 is second

    def test_top_k_indices_empty_when_no_importances(self):
        sr = self._make_step_report()
        assert sr.top_k_indices(k=5) == []

    def test_top_k_respects_k_bound(self):
        sr = self._make_step_report(importances=np.array([0.2, 0.8]))
        # k > len(importances) should not raise
        assert len(sr.top_k_indices(k=10)) == 2

    def test_repr_contains_name(self):
        sr = self._make_step_report()
        assert "MyStep" in repr(sr)


# ===========================================================================
# Section 2 — PipelineReport
# ===========================================================================

class TestPipelineReport:

    def _make_report_with_importances(self, imp_values):
        imp = np.array(imp_values)
        step = StepReport(
            step_id="x", step_name="RF", step_type="classifier",
            operation_class="RandomForestClassifier", params={},
            feature_importances=imp, importance_source="feature_importances_",
        )
        report = PipelineReport(pipeline_name="Test", n_steps=1)
        report.steps.append(step)
        return report

    def test_final_step_returns_last(self):
        report = self._make_report_with_importances([0.5, 0.5])
        assert report.final_step() is not None
        assert report.final_step().step_name == "RF"

    def test_final_step_none_on_empty(self):
        report = PipelineReport(pipeline_name="Empty", n_steps=0)
        assert report.final_step() is None

    def test_predictive_step_prefers_clf_regressor(self):
        report = PipelineReport(pipeline_name="P", n_steps=2)
        # scaler step without importances
        sc_step = StepReport(
            step_id="sc", step_name="Scaler", step_type="scaler",
            operation_class="StandardScaler", params={},
        )
        # classifier step with importances
        clf_step = StepReport(
            step_id="clf", step_name="RF", step_type="classifier",
            operation_class="RF", params={},
            feature_importances=np.array([0.4, 0.6]),
            importance_source="feature_importances_",
        )
        report.steps = [sc_step, clf_step]
        assert report.predictive_step() is clf_step

    def test_predictive_step_none_when_no_importances(self):
        report = PipelineReport(pipeline_name="P", n_steps=1)
        sc_step = StepReport(
            step_id="sc", step_name="Scaler", step_type="scaler",
            operation_class="StandardScaler", params={},
        )
        report.steps = [sc_step]
        assert report.predictive_step() is None

    def test_top_features_returns_correct_length(self):
        report = self._make_report_with_importances([0.1, 0.5, 0.3, 0.05, 0.05])
        names = ["a", "b", "c", "d", "e"]
        top = report.top_features(feature_names=names, k=3)
        assert len(top) == 3

    def test_top_features_ordered_by_importance(self):
        report = self._make_report_with_importances([0.1, 0.5, 0.3, 0.05, 0.05])
        names = ["a", "b", "c", "d", "e"]
        top = report.top_features(feature_names=names, k=5)
        scores = [w for _, w in top]
        assert scores == sorted(scores, reverse=True)

    def test_top_features_uses_generic_names_when_none(self):
        report = self._make_report_with_importances([0.4, 0.6])
        top = report.top_features(feature_names=None, k=2)
        names = [n for n, _ in top]
        assert all(n.startswith("feature_") for n in names)

    def test_top_features_empty_when_no_importances(self):
        report = PipelineReport(pipeline_name="P", n_steps=0)
        assert report.top_features() == []

    def test_summary_contains_pipeline_name(self, simple_clf_pipeline):
        introspector = PipelineIntrospector()
        report = introspector.inspect(simple_clf_pipeline)
        summary = report.summary()
        assert "SimpleCLF" in summary

    def test_summary_contains_all_step_types(self, simple_clf_pipeline):
        introspector = PipelineIntrospector()
        report = introspector.inspect(simple_clf_pipeline)
        summary = report.summary()
        assert "scaler" in summary
        assert "classifier" in summary

    def test_repr_contains_score(self, dummy_fitness, simple_clf_pipeline):
        introspector = PipelineIntrospector()
        report = introspector.inspect(simple_clf_pipeline, fitness=dummy_fitness)
        assert "0.9400" in repr(report)


# ===========================================================================
# Section 3 — PipelineIntrospector
# ===========================================================================

class TestPipelineIntrospector:

    def test_inspect_returns_pipeline_report(self, simple_clf_pipeline):
        report = PipelineIntrospector().inspect(simple_clf_pipeline)
        assert isinstance(report, PipelineReport)

    def test_step_count_matches(self, simple_clf_pipeline):
        report = PipelineIntrospector().inspect(simple_clf_pipeline)
        assert report.n_steps == len(simple_clf_pipeline)
        assert len(report.steps) == len(simple_clf_pipeline)

    def test_unfitted_pipeline_has_no_importances(self, simple_clf_pipeline):
        report = PipelineIntrospector().inspect(simple_clf_pipeline)
        for step in report.steps:
            assert not step.has_importances()

    def test_fitted_rf_has_feature_importances(self, fitted_rf_pipeline):
        report = PipelineIntrospector().inspect(fitted_rf_pipeline)
        pred_step = report.predictive_step()
        assert pred_step is not None
        assert pred_step.has_importances()
        assert pred_step.importance_source == "feature_importances_"

    def test_fitted_lr_has_coef_importances(self, fitted_lr_pipeline):
        report = PipelineIntrospector().inspect(fitted_lr_pipeline)
        pred_step = report.predictive_step()
        assert pred_step is not None
        assert pred_step.has_importances()
        assert pred_step.importance_source == "coef_"

    def test_importances_sum_to_one(self, fitted_rf_pipeline):
        report = PipelineIntrospector().inspect(fitted_rf_pipeline)
        pred_step = report.predictive_step()
        assert abs(pred_step.feature_importances.sum() - 1.0) < 1e-6

    def test_importances_non_negative(self, fitted_rf_pipeline):
        report = PipelineIntrospector().inspect(fitted_rf_pipeline)
        pred_step = report.predictive_step()
        assert np.all(pred_step.feature_importances >= 0)

    def test_n_input_features_populated_after_fit(self, fitted_rf_pipeline):
        report = PipelineIntrospector().inspect(fitted_rf_pipeline)
        for step in report.steps:
            if step.step_type in ("classifier", "regressor"):
                assert step.n_input_features is not None

    def test_fitness_carried_through(self, simple_clf_pipeline, dummy_fitness):
        report = PipelineIntrospector().inspect(
            simple_clf_pipeline, fitness=dummy_fitness
        )
        assert report.fitness is dummy_fitness

    def test_pca_step_extracts_explained_variance(self, iris_data):
        """PCA should surface explained_variance_ratio_ as importances."""
        X, y = iris_data
        p = Pipeline(name="PCATest")
        sc = PipelineStep(
            name="StandardScaler", step_type="scaler",
            operation=StandardScaler(),
            input_type=CleanTabular, output_type=ScaledData,
        )
        pca = PipelineStep(
            name="PCA", step_type="dim_reducer",
            operation=PCA(n_components=2),
            input_type=ScaledData, output_type=Features,
        )
        p.add_step(sc)
        p.add_step(pca)
        p.add_edge(sc.id, pca.id)

        # Fit just the PCA step manually (pipeline.fit needs a sink with predict)
        sc.operation.fit(X)
        X_sc = sc.operation.transform(X)
        pca.operation.fit(X_sc)

        report = PipelineIntrospector().inspect(p)
        pca_report = next(s for s in report.steps if s.step_type == "dim_reducer")
        assert pca_report.has_importances()
        assert pca_report.importance_source == "explained_variance_ratio_"

    def test_top_features_with_iris_names(
        self, fitted_rf_pipeline, iris_feature_names
    ):
        report = PipelineIntrospector().inspect(fitted_rf_pipeline)
        top = report.top_features(feature_names=iris_feature_names, k=4)
        assert len(top) == 4
        feature_names_returned = [n for n, _ in top]
        for name in feature_names_returned:
            assert name in iris_feature_names


# ===========================================================================
# Section 4 — PipelineStory
# ===========================================================================

class TestPipelineStory:

    def test_narrate_returns_string(self):
        log = _make_log([0.70, 0.75, 0.80])
        story = PipelineStory(log, task="classification", metric="accuracy")
        text = story.narrate()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_narrate_contains_task_and_metric(self):
        log = _make_log([0.80, 0.85])
        story = PipelineStory(log, task="classification", metric="accuracy")
        text = story.narrate()
        assert "classification" in text
        assert "accuracy" in text

    def test_narrate_contains_generation_count(self):
        log = _make_log([0.80, 0.85, 0.87])
        story = PipelineStory(log)
        text = story.narrate()
        assert "3" in text  # 3 generations

    def test_narrate_empty_log(self):
        log = EvolutionLog()
        story = PipelineStory(log)
        text = story.narrate()
        assert "No generations" in text

    def test_narrate_marks_improvement_generations(self):
        log = _make_log([0.70, 0.70, 0.75, 0.75, 0.80])
        story = PipelineStory(log)
        text = story.narrate()
        assert "new best" in text

    def test_narrate_detects_plateau(self):
        # Last 4 generations have same score → plateau
        log = _make_log([0.70, 0.80, 0.80, 0.80, 0.80])
        story = PipelineStory(log)
        text = story.narrate()
        assert "stable" in text.lower() or "converge" in text.lower()

    def test_narrate_includes_pipeline_summary_when_provided(
        self, simple_clf_pipeline
    ):
        log = _make_log([0.80, 0.85])
        story = PipelineStory(log, best_pipeline=simple_clf_pipeline)
        text = story.narrate()
        assert "SimpleCLF" in text or "StandardScaler" in text

    def test_pipeline_summary_contains_step_types(self, simple_clf_pipeline):
        log = _make_log([0.80])
        story = PipelineStory(log)
        summary = story.pipeline_summary(simple_clf_pipeline)
        assert "scaler" in summary
        assert "classifier" in summary

    def test_pipeline_summary_contains_pipeline_name(self, simple_clf_pipeline):
        log = _make_log([0.80])
        story = PipelineStory(log)
        summary = story.pipeline_summary(simple_clf_pipeline)
        assert "SimpleCLF" in summary

    def test_generation_table_has_header(self):
        log = _make_log([0.70, 0.80, 0.85])
        story = PipelineStory(log)
        table = story.generation_table()
        assert "Gen" in table
        assert "Best" in table
        assert "Mean" in table

    def test_generation_table_row_count(self):
        log = _make_log([0.70, 0.80, 0.85])
        story = PipelineStory(log)
        table = story.generation_table()
        # header + separator + 3 data rows = 5 lines minimum
        lines = [l for l in table.splitlines() if l.strip()]
        assert len(lines) >= 5

    def test_generation_table_scores_appear(self):
        log = _make_log([0.7123, 0.8456])
        story = PipelineStory(log)
        table = story.generation_table()
        assert "0.7123" in table
        assert "0.8456" in table

    def test_repr(self):
        log = _make_log([0.80])
        story = PipelineStory(log, task="regression", metric="r2")
        r = repr(story)
        assert "regression" in r
        assert "r2" in r


# ===========================================================================
# Section 5 — Visualiser smoke tests
# ===========================================================================

class TestVisualizer:
    """Smoke tests: verify the functions return a Figure without crashing.
    We don't assert pixel-level content — just that the call succeeds."""

    def test_render_pipeline_returns_figure(self, simple_clf_pipeline):
        pytest.importorskip("matplotlib")
        from c60.explainability.visualizer import render_pipeline
        import matplotlib.pyplot as plt
        fig = render_pipeline(simple_clf_pipeline)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_render_pipeline_empty_pipeline(self):
        pytest.importorskip("matplotlib")
        from c60.explainability.visualizer import render_pipeline
        import matplotlib.pyplot as plt
        empty = Pipeline(name="empty")
        fig = render_pipeline(empty)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_render_evolution_returns_figure(self):
        pytest.importorskip("matplotlib")
        from c60.explainability.visualizer import render_evolution
        import matplotlib.pyplot as plt
        log = _make_log([0.70, 0.75, 0.80, 0.82])
        fig = render_evolution(log)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_render_evolution_empty_log(self):
        pytest.importorskip("matplotlib")
        from c60.explainability.visualizer import render_evolution
        import matplotlib.pyplot as plt
        fig = render_evolution(EvolutionLog())
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_render_pipeline_custom_title(self, simple_clf_pipeline):
        pytest.importorskip("matplotlib")
        from c60.explainability.visualizer import render_pipeline
        import matplotlib.pyplot as plt
        fig = render_pipeline(simple_clf_pipeline, title="My Custom Title")
        assert hasattr(fig, "savefig")
        plt.close(fig)
