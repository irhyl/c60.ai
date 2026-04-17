"""
test_hybrid.py — Tests for NeuralAutoencoder, NeuralClassifier,
                  register_hybrid_nodes(), and end-to-end hybrid pipeline runs.

All tests are skipped if torch is not installed (pytest.importorskip).

Structure
---------
Section 1 — EmbeddedData type compatibility
Section 2 — NeuralAutoencoder (sklearn API + correctness)
Section 3 — NeuralClassifier  (sklearn API + correctness)
Section 4 — deepcopy safety   (Pipeline.clone() path)
Section 5 — Hybrid Pipeline   (nodes inside a typed DAG)
Section 6 — Registry extension (register_hybrid_nodes)
Section 7 — FitnessEvaluator integration (end-to-end CV)
"""

import copy

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Skip entire module if torch is absent
torch = pytest.importorskip("torch")

from c60.core.pipeline import Pipeline, PipelineStep
from c60.core.registry import OperationRegistry
from c60.core.types import (
    CleanTabular, EmbeddedData, ScaledData, ClassLabels, is_compatible,
)
from c60.evaluation.fitness import FitnessEvaluator
from c60.hybrid.node import NeuralAutoencoder, NeuralClassifier
from c60.hybrid.registry_ext import register_hybrid_nodes


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture
def X_small():
    """Tiny (30, 8) float array — fast to train on."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((30, 8)).astype(np.float32)


@pytest.fixture
def X_y_small():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 6)).astype(np.float32)
    y = rng.integers(0, 3, size=40)
    return X, y


@pytest.fixture
def autoencoder():
    return NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4, epochs=3, random_state=0)


@pytest.fixture
def classifier():
    return NeuralClassifier(hidden_dim=16, epochs=5, random_state=0)


@pytest.fixture
def hybrid_registry():
    """Fresh registry with hybrid nodes registered."""
    reg = OperationRegistry()
    register_hybrid_nodes(reg)
    return reg


# ===========================================================================
# Section 1 — EmbeddedData type compatibility
# ===========================================================================

class TestEmbeddedDataType:

    def test_embedded_is_subclass_of_scaled(self):
        assert issubclass(EmbeddedData, ScaledData)

    def test_is_compatible_embedded_to_scaled(self):
        # EmbeddedData output → ScaledData-accepting classifier: valid
        assert is_compatible(EmbeddedData, ScaledData)

    def test_is_compatible_embedded_to_clean_tabular(self):
        from c60.core.types import CleanTabular
        assert is_compatible(EmbeddedData, CleanTabular)

    def test_is_not_compatible_scaled_to_embedded(self):
        # ScaledData is NOT a subtype of EmbeddedData
        assert not is_compatible(ScaledData, EmbeddedData)

    def test_type_name_registered(self):
        from c60.core.types import TYPE_NAMES
        assert EmbeddedData in TYPE_NAMES
        assert TYPE_NAMES[EmbeddedData] == "EmbeddedData"


# ===========================================================================
# Section 2 — NeuralAutoencoder
# ===========================================================================

class TestNeuralAutoencoder:

    def test_fit_returns_self(self, autoencoder, X_small):
        result = autoencoder.fit(X_small)
        assert result is autoencoder

    def test_n_features_in_set_after_fit(self, autoencoder, X_small):
        autoencoder.fit(X_small)
        assert autoencoder.n_features_in_ == X_small.shape[1]

    def test_reconstruction_loss_set_after_fit(self, autoencoder, X_small):
        autoencoder.fit(X_small)
        assert autoencoder.reconstruction_loss_ is not None
        assert autoencoder.reconstruction_loss_ >= 0.0

    def test_transform_output_shape(self, autoencoder, X_small):
        autoencoder.fit(X_small)
        Z = autoencoder.transform(X_small)
        assert Z.shape == (X_small.shape[0], autoencoder.bottleneck_dim)

    def test_transform_output_dtype_float(self, autoencoder, X_small):
        autoencoder.fit(X_small)
        Z = autoencoder.transform(X_small)
        assert Z.dtype in (np.float32, np.float64)

    def test_fit_transform_matches_fit_then_transform(self, X_small):
        ae1 = NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4, epochs=3, random_state=7)
        ae2 = NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4, epochs=3, random_state=7)
        Z1 = ae1.fit_transform(X_small)
        ae2.fit(X_small)
        Z2 = ae2.transform(X_small)
        np.testing.assert_allclose(Z1, Z2, rtol=1e-5)

    def test_transform_before_fit_raises(self, X_small):
        ae = NeuralAutoencoder()
        with pytest.raises(RuntimeError, match="fitted"):
            ae.transform(X_small)

    def test_get_params_keys(self, autoencoder):
        params = autoencoder.get_params()
        for key in ("hidden_dim", "bottleneck_dim", "epochs", "lr",
                    "batch_size", "random_state"):
            assert key in params

    def test_set_params_updates_attrs(self, autoencoder):
        autoencoder.set_params(hidden_dim=16, epochs=10)
        assert autoencoder.hidden_dim == 16
        assert autoencoder.epochs == 10

    def test_set_params_invalidates_model(self, autoencoder, X_small):
        autoencoder.fit(X_small)
        assert autoencoder._encoder is not None
        autoencoder.set_params(hidden_dim=4)
        assert autoencoder._encoder is None

    def test_repr_contains_class_name(self, autoencoder):
        assert "NeuralAutoencoder" in repr(autoencoder)

    def test_deterministic_with_same_seed(self, X_small):
        ae1 = NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4, epochs=5, random_state=99)
        ae2 = NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4, epochs=5, random_state=99)
        Z1 = ae1.fit_transform(X_small)
        Z2 = ae2.fit_transform(X_small)
        np.testing.assert_allclose(Z1, Z2, rtol=1e-5)

    def test_different_seeds_give_different_results(self, X_small):
        ae1 = NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4, epochs=10, random_state=1)
        ae2 = NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4, epochs=10, random_state=2)
        Z1 = ae1.fit_transform(X_small)
        Z2 = ae2.fit_transform(X_small)
        assert not np.allclose(Z1, Z2)

    def test_bottleneck_reduces_dimensionality(self, X_small):
        ae = NeuralAutoencoder(hidden_dim=8, bottleneck_dim=3, epochs=3, random_state=0)
        Z = ae.fit_transform(X_small)
        assert Z.shape[1] == 3
        assert Z.shape[1] < X_small.shape[1]


# ===========================================================================
# Section 3 — NeuralClassifier
# ===========================================================================

class TestNeuralClassifier:

    def test_fit_returns_self(self, classifier, X_y_small):
        X, y = X_y_small
        result = classifier.fit(X, y)
        assert result is classifier

    def test_n_features_in_set_after_fit(self, classifier, X_y_small):
        X, y = X_y_small
        classifier.fit(X, y)
        assert classifier.n_features_in_ == X.shape[1]

    def test_classes_set_after_fit(self, classifier, X_y_small):
        X, y = X_y_small
        classifier.fit(X, y)
        assert classifier.classes_ is not None
        assert len(classifier.classes_) == len(np.unique(y))

    def test_predict_output_shape(self, classifier, X_y_small):
        X, y = X_y_small
        classifier.fit(X, y)
        preds = classifier.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_returns_valid_labels(self, classifier, X_y_small):
        X, y = X_y_small
        classifier.fit(X, y)
        preds = classifier.predict(X)
        valid = set(np.unique(y))
        assert all(p in valid for p in preds)

    def test_predict_proba_shape(self, classifier, X_y_small):
        X, y = X_y_small
        classifier.fit(X, y)
        proba = classifier.predict_proba(X)
        n_classes = len(np.unique(y))
        assert proba.shape == (len(X), n_classes)

    def test_predict_proba_sums_to_one(self, classifier, X_y_small):
        X, y = X_y_small
        classifier.fit(X, y)
        proba = classifier.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(X)), atol=1e-5)

    def test_predict_before_fit_raises(self, X_y_small):
        clf = NeuralClassifier()
        X, _ = X_y_small
        with pytest.raises(RuntimeError, match="fitted"):
            clf.predict(X)

    def test_get_params_keys(self, classifier):
        params = classifier.get_params()
        for key in ("hidden_dim", "epochs", "lr", "batch_size", "random_state"):
            assert key in params

    def test_set_params_updates_attrs(self, classifier):
        classifier.set_params(hidden_dim=32, epochs=20)
        assert classifier.hidden_dim == 32
        assert classifier.epochs == 20

    def test_set_params_invalidates_model(self, classifier, X_y_small):
        X, y = X_y_small
        classifier.fit(X, y)
        assert classifier._net is not None
        classifier.set_params(hidden_dim=8)
        assert classifier._net is None

    def test_repr_contains_class_name(self, classifier):
        assert "NeuralClassifier" in repr(classifier)

    def test_trains_to_nonzero_accuracy(self, iris):
        """After 20 epochs on iris the classifier should beat random (>40%)."""
        X, y = iris
        X = StandardScaler().fit_transform(X)
        clf = NeuralClassifier(hidden_dim=32, epochs=20, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.40, f"Expected >40% train accuracy, got {acc:.2%}"


# ===========================================================================
# Section 4 — deepcopy safety
# ===========================================================================

class TestDeepcopy:

    def test_autoencoder_deepcopy_unfitted(self, autoencoder):
        ae2 = copy.deepcopy(autoencoder)
        assert ae2 is not autoencoder
        assert ae2._encoder is None

    def test_autoencoder_deepcopy_fitted(self, X_small):
        ae = NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4, epochs=3, random_state=0)
        ae.fit(X_small)
        ae2 = copy.deepcopy(ae)
        assert ae2._encoder is not None
        # Independent: mutate copy does not affect original
        Z_orig = ae.transform(X_small)
        Z_copy = ae2.transform(X_small)
        np.testing.assert_allclose(Z_orig, Z_copy, rtol=1e-5)

    def test_autoencoder_copy_is_independent(self, X_small):
        ae = NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4, epochs=3, random_state=0)
        ae.fit(X_small)
        ae2 = copy.deepcopy(ae)
        ae2.set_params(hidden_dim=16)
        assert ae.hidden_dim == 8   # original unchanged

    def test_classifier_deepcopy_unfitted(self, classifier):
        clf2 = copy.deepcopy(classifier)
        assert clf2 is not classifier
        assert clf2._net is None

    def test_classifier_deepcopy_fitted(self, X_y_small):
        X, y = X_y_small
        clf = NeuralClassifier(hidden_dim=16, epochs=5, random_state=0)
        clf.fit(X, y)
        clf2 = copy.deepcopy(clf)
        assert clf2._net is not None
        preds1 = clf.predict(X)
        preds2 = clf2.predict(X)
        np.testing.assert_array_equal(preds1, preds2)

    def test_classifier_copy_is_independent(self, X_y_small):
        X, y = X_y_small
        clf = NeuralClassifier(hidden_dim=16, epochs=5, random_state=0)
        clf.fit(X, y)
        clf2 = copy.deepcopy(clf)
        clf2.set_params(hidden_dim=8)
        assert clf.hidden_dim == 16  # original unchanged


# ===========================================================================
# Section 5 — Hybrid Pipeline (nodes inside the typed DAG)
# ===========================================================================

class TestHybridPipeline:

    def _make_autoencoder_pipeline(self):
        """StandardScaler → NeuralAutoencoder → LogisticRegression"""
        from sklearn.linear_model import LogisticRegression
        from c60.core.types import ScaledData, EmbeddedData, ClassLabels

        p = Pipeline(name="AE_Pipeline")
        sc = PipelineStep(
            name="StandardScaler", step_type="scaler",
            operation=StandardScaler(),
            input_type=CleanTabular, output_type=ScaledData,
        )
        ae = PipelineStep(
            name="NeuralAutoencoder", step_type="dim_reducer",
            operation=NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4,
                                        epochs=3, random_state=0),
            input_type=ScaledData, output_type=EmbeddedData,
        )
        clf = PipelineStep(
            name="LogisticRegression", step_type="classifier",
            operation=LogisticRegression(max_iter=200),
            input_type=ScaledData,   # EmbeddedData IS-A ScaledData → valid
            output_type=ClassLabels,
        )
        p.add_step(sc)
        p.add_step(ae)
        p.add_step(clf)
        p.add_edge(sc.id, ae.id)
        p.add_edge(ae.id, clf.id)
        return p

    def _make_neural_classifier_pipeline(self):
        """StandardScaler → NeuralClassifier"""
        from c60.core.types import ScaledData, ClassLabels

        p = Pipeline(name="NC_Pipeline")
        sc = PipelineStep(
            name="StandardScaler", step_type="scaler",
            operation=StandardScaler(),
            input_type=CleanTabular, output_type=ScaledData,
        )
        clf = PipelineStep(
            name="NeuralClassifier", step_type="classifier",
            operation=NeuralClassifier(hidden_dim=16, epochs=5, random_state=0),
            input_type=ScaledData, output_type=ClassLabels,
        )
        p.add_step(sc)
        p.add_step(clf)
        p.add_edge(sc.id, clf.id)
        return p

    def test_autoencoder_pipeline_type_edge_valid(self):
        """EmbeddedData → ScaledData edge must pass type check."""
        p = self._make_autoencoder_pipeline()
        from c60.execution.validator import PipelineValidator
        vr = PipelineValidator().validate(p)
        assert vr.is_valid, vr.errors

    def test_autoencoder_pipeline_fit_predict(self, iris):
        X, y = iris
        p = self._make_autoencoder_pipeline()
        p.fit(X, y)
        preds = p.predict(X)
        assert len(preds) == len(y)

    def test_autoencoder_pipeline_accuracy_above_chance(self, iris):
        X, y = iris
        p = self._make_autoencoder_pipeline()
        p.fit(X, y)
        preds = p.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.33, f"Accuracy {acc:.2%} below chance (3 classes)"

    def test_neural_classifier_pipeline_fit_predict(self, iris):
        X, y = iris
        p = self._make_neural_classifier_pipeline()
        p.fit(X, y)
        preds = p.predict(X)
        assert len(preds) == len(y)

    def test_pipeline_clone_works(self, iris):
        X, y = iris
        p = self._make_autoencoder_pipeline()
        p.fit(X, y)
        p2 = p.clone()
        # Clone should be independently fittable
        p2.fit(X, y)
        preds = p2.predict(X)
        assert len(preds) == len(y)

    def test_pipeline_structure_hash_stable(self):
        p = self._make_autoencoder_pipeline()
        h1 = p.structure_hash()
        h2 = p.clone().structure_hash()
        assert h1 == h2  # clone has same structure → same hash

    def test_fully_neural_pipeline(self, iris):
        """StandardScaler → NeuralAutoencoder → NeuralClassifier"""
        from c60.core.types import EmbeddedData, ClassLabels
        X, y = iris
        p = Pipeline(name="FullNeural")
        sc = PipelineStep(
            name="StandardScaler", step_type="scaler",
            operation=StandardScaler(),
            input_type=CleanTabular, output_type=ScaledData,
        )
        ae = PipelineStep(
            name="NeuralAutoencoder", step_type="dim_reducer",
            operation=NeuralAutoencoder(hidden_dim=8, bottleneck_dim=4,
                                        epochs=3, random_state=0),
            input_type=ScaledData, output_type=EmbeddedData,
        )
        nc = PipelineStep(
            name="NeuralClassifier", step_type="classifier",
            operation=NeuralClassifier(hidden_dim=16, epochs=5, random_state=0),
            input_type=ScaledData,   # EmbeddedData IS-A ScaledData
            output_type=ClassLabels,
        )
        p.add_step(sc); p.add_step(ae); p.add_step(nc)
        p.add_edge(sc.id, ae.id)
        p.add_edge(ae.id, nc.id)
        p.fit(X, y)
        preds = p.predict(X)
        assert len(preds) == len(y)


# ===========================================================================
# Section 6 — Registry extension
# ===========================================================================

class TestRegistryExtension:

    def test_register_returns_true_when_torch_available(self):
        reg = OperationRegistry()
        result = register_hybrid_nodes(reg)
        assert result is True

    def test_autoencoder_in_registry(self, hybrid_registry):
        specs = hybrid_registry.get_by_type("dim_reducer")
        class_names = [s.op_class.__name__ for s in specs]
        assert "NeuralAutoencoder" in class_names

    def test_classifier_in_registry(self, hybrid_registry):
        specs = hybrid_registry.get_by_type("classifier")
        class_names = [s.op_class.__name__ for s in specs]
        assert "NeuralClassifier" in class_names

    def test_autoencoder_spec_types(self, hybrid_registry):
        specs = hybrid_registry.get_by_type("dim_reducer")
        ae_specs = [s for s in specs if s.op_class.__name__ == "NeuralAutoencoder"]
        assert len(ae_specs) == 1
        spec = ae_specs[0]
        assert spec.input_type is ScaledData
        assert spec.output_type is EmbeddedData

    def test_classifier_spec_types(self, hybrid_registry):
        specs = hybrid_registry.get_by_type("classifier")
        nc_specs = [s for s in specs if s.op_class.__name__ == "NeuralClassifier"]
        assert len(nc_specs) == 1
        spec = nc_specs[0]
        assert spec.input_type is ScaledData
        assert spec.output_type is ClassLabels

    def test_autoencoder_search_space_has_expected_params(self, hybrid_registry):
        specs = hybrid_registry.get_by_type("dim_reducer")
        ae_spec = next(s for s in specs if s.op_class.__name__ == "NeuralAutoencoder")
        for param in ("hidden_dim", "bottleneck_dim", "epochs", "lr"):
            assert param in ae_spec.search_space

    def test_sample_params_returns_valid_instance(self, hybrid_registry):
        specs = hybrid_registry.get_by_type("dim_reducer")
        ae_spec = next(s for s in specs if s.op_class.__name__ == "NeuralAutoencoder")
        params = ae_spec.sample_params()
        instance = ae_spec.instantiate(params)
        assert isinstance(instance, NeuralAutoencoder)

    def test_register_idempotent(self):
        """Calling register_hybrid_nodes twice should not raise."""
        reg = OperationRegistry()
        register_hybrid_nodes(reg)
        register_hybrid_nodes(reg)   # second call is a no-op (duplicate key)


# ===========================================================================
# Section 7 — FitnessEvaluator integration
# ===========================================================================

class TestFitnessEvaluatorIntegration:

    def _make_neural_pipeline(self):
        from c60.core.types import ScaledData, ClassLabels
        p = Pipeline(name="NeuralFit")
        sc = PipelineStep(
            name="StandardScaler", step_type="scaler",
            operation=StandardScaler(),
            input_type=CleanTabular, output_type=ScaledData,
        )
        clf = PipelineStep(
            name="NeuralClassifier", step_type="classifier",
            operation=NeuralClassifier(hidden_dim=32, epochs=30, random_state=0),
            input_type=ScaledData, output_type=ClassLabels,
        )
        p.add_step(sc); p.add_step(clf)
        p.add_edge(sc.id, clf.id)
        return p

    def test_evaluator_runs_without_error(self, iris):
        X, y = iris
        evaluator = FitnessEvaluator(
            metric="accuracy", cv=2, task="classification",
            timeout_seconds=60.0, complexity_penalty=0.0,
        )
        p = self._make_neural_pipeline()
        result = evaluator.evaluate(p, X, y)
        assert result is not None
        assert not result.failed, result.error

    def test_evaluator_returns_reasonable_score(self, iris):
        X, y = iris
        evaluator = FitnessEvaluator(
            metric="accuracy", cv=2, task="classification",
            timeout_seconds=60.0, complexity_penalty=0.0,
        )
        p = self._make_neural_pipeline()
        result = evaluator.evaluate(p, X, y)
        # 2-fold CV on iris: even a bad model should beat random (>0.33)
        assert result.raw_score > 0.33, (
            f"Expected > 33% accuracy, got {result.raw_score:.2%}"
        )

    def test_cache_hit_on_identical_pipeline(self, iris):
        from c60.evaluation.cache import EvaluationCache
        X, y = iris
        cache = EvaluationCache()
        evaluator = FitnessEvaluator(
            metric="accuracy", cv=2, task="classification",
            timeout_seconds=60.0, complexity_penalty=0.0,
            cache=cache,
        )
        p = self._make_neural_pipeline()
        evaluator.evaluate(p, X, y)
        evaluator.evaluate(p.clone(), X, y)   # same structure → cache hit
        assert cache.hits >= 1
