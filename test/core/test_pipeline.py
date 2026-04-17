"""
test_pipeline.py — Tests for core pipeline data structures.

Coverage:
  types.py        — DataType lattice, is_compatible, compatible_input_types
  pipeline.py     — PipelineStep construction, equality, cloning, serialization
                    Pipeline add/query/execute, edge validation, cycle detection
  registry.py     — OperationRegistry lookups, sampling, registration
"""

import copy
import uuid

import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from c60.core.types import (
    DataType, TabularData,
    RawTabular, CleanTabular, ScaledData, UnscaledData,
    Features, Predictions, ClassLabels, RegressionValues, Probabilities,
    is_compatible, compatible_input_types, type_name,
)
from c60.core.pipeline import PipelineStep, Pipeline
from c60.core.registry import (
    OperationRegistry, OperationSpec, IntRange, FloatRange, Categorical, Boolean,
    default_registry,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def scaler_step():
    return PipelineStep(
        name="Scaler",
        step_type="scaler",
        operation=StandardScaler(),
        input_type=CleanTabular,
        output_type=ScaledData,
    )


@pytest.fixture
def classifier_step():
    return PipelineStep(
        name="Classifier",
        step_type="classifier",
        operation=LogisticRegression(max_iter=200),
        input_type=ScaledData,
        output_type=ClassLabels,
    )


@pytest.fixture
def linear_pipeline(scaler_step, classifier_step):
    """A minimal valid pipeline: Scaler → Classifier."""
    p = Pipeline(name="LinearPipeline")
    p.add_step(scaler_step)
    p.add_step(classifier_step)
    p.add_edge(scaler_step.id, classifier_step.id)
    return p


@pytest.fixture
def iris():
    return load_iris(return_X_y=True)


# ===========================================================================
# Section 1 — Type lattice
# ===========================================================================

class TestTypeCompatibility:

    def test_exact_type_is_compatible_with_itself(self):
        assert is_compatible(ScaledData, ScaledData)
        assert is_compatible(CleanTabular, CleanTabular)
        assert is_compatible(ClassLabels, ClassLabels)

    def test_subtype_is_compatible_with_parent(self):
        # ScaledData IS-A CleanTabular IS-A TabularData IS-A DataType
        assert is_compatible(ScaledData, CleanTabular)
        assert is_compatible(ScaledData, TabularData)
        assert is_compatible(ScaledData, DataType)
        assert is_compatible(ClassLabels, Predictions)
        assert is_compatible(ClassLabels, DataType)
        assert is_compatible(RegressionValues, Predictions)

    def test_parent_not_compatible_with_subtype(self):
        # CleanTabular is NOT ScaledData
        assert not is_compatible(CleanTabular, ScaledData)
        assert not is_compatible(Predictions, ClassLabels)
        assert not is_compatible(DataType, Features)

    def test_unrelated_types_not_compatible(self):
        assert not is_compatible(ClassLabels, Features)
        assert not is_compatible(ScaledData, Probabilities)
        assert not is_compatible(RegressionValues, ClassLabels)
        assert not is_compatible(Features, RawTabular)

    def test_compatible_input_types_includes_ancestors(self):
        result = compatible_input_types(ScaledData)
        assert ScaledData in result
        assert CleanTabular in result
        assert TabularData in result
        assert DataType in result
        # Should NOT contain unrelated types
        assert Features not in result
        assert ClassLabels not in result

    def test_type_name_returns_string(self):
        assert type_name(ScaledData) == "ScaledData"
        assert type_name(ClassLabels) == "ClassLabels"
        assert type_name(DataType) == "Data"


# ===========================================================================
# Section 2 — PipelineStep
# ===========================================================================

class TestPipelineStep:

    def test_default_initialization(self):
        step = PipelineStep()
        assert isinstance(step.id, str)
        assert len(step.id) == 36          # UUID4 format
        assert step.name == "UnnamedStep"
        assert step.step_type == "generic"
        assert step.operation is None
        assert step.params == {}
        assert step.input_type is CleanTabular
        assert step.output_type is CleanTabular

    def test_explicit_initialization(self):
        op = StandardScaler()
        step = PipelineStep(
            step_id="abc-123",
            name="MyScaler",
            step_type="scaler",
            operation=op,
            params={"with_mean": True},
            input_type=CleanTabular,
            output_type=ScaledData,
        )
        assert step.id == "abc-123"
        assert step.name == "MyScaler"
        assert step.step_type == "scaler"
        assert step.operation is op
        assert step.params == {"with_mean": True}
        assert step.input_type is CleanTabular
        assert step.output_type is ScaledData

    def test_equality_based_on_id(self):
        shared_id = str(uuid.uuid4())
        s1 = PipelineStep(step_id=shared_id, name="A")
        s2 = PipelineStep(step_id=shared_id, name="B")   # same id, different name
        s3 = PipelineStep(name="C")                       # different id
        assert s1 == s2
        assert s1 != s3
        assert s2 != s3

    def test_equality_with_non_step_returns_not_implemented(self):
        step = PipelineStep(name="X")
        result = step.__eq__("not_a_step")
        assert result is NotImplemented

    def test_hash_consistent_with_equality(self):
        shared_id = str(uuid.uuid4())
        s1 = PipelineStep(step_id=shared_id)
        s2 = PipelineStep(step_id=shared_id)
        s3 = PipelineStep()
        assert hash(s1) == hash(s2)
        # Steps with same id are deduplicated in a set
        assert len({s1, s2}) == 1
        assert len({s1, s3}) == 2

    def test_repr_contains_key_info(self):
        step = PipelineStep(name="TestStep", step_type="scaler",
                            output_type=ScaledData)
        r = repr(step)
        assert "TestStep" in r
        assert "scaler" in r
        assert "ScaledData" in r

    # -----------------------------------------------------------------------
    # clone()
    # -----------------------------------------------------------------------

    def test_clone_produces_independent_copy(self):
        original = PipelineStep(
            name="Scaler",
            step_type="scaler",
            operation=StandardScaler(),
            params={"with_mean": True},
            input_type=CleanTabular,
            output_type=ScaledData,
        )
        cloned = original.clone()

        # Different id (it's a new node)
        assert cloned.id != original.id

        # Structural properties preserved
        assert cloned.name == original.name
        assert cloned.step_type == original.step_type
        assert cloned.input_type is original.input_type
        assert cloned.output_type is original.output_type

        # Mutating clone does not affect original
        cloned.params["with_mean"] = False
        assert original.params["with_mean"] is True

    def test_clone_deep_copies_operation(self):
        op = StandardScaler()
        original = PipelineStep(operation=op)
        cloned = original.clone()
        assert cloned.operation is not op   # different object

    # -----------------------------------------------------------------------
    # to_dict()
    # -----------------------------------------------------------------------

    def test_to_dict_structure(self):
        step = PipelineStep(
            step_id="xyz",
            name="Scaler",
            step_type="scaler",
            operation=StandardScaler(),
            params={"with_mean": True},
            input_type=CleanTabular,
            output_type=ScaledData,
        )
        d = step.to_dict()
        assert d["id"] == "xyz"
        assert d["name"] == "Scaler"
        assert d["step_type"] == "scaler"
        assert d["params"] == {"with_mean": True}
        assert d["input_type"] == "CleanTabular"
        assert d["output_type"] == "ScaledData"
        assert "StandardScaler" in d["operation_class"]

    def test_to_dict_none_operation(self):
        step = PipelineStep(name="Empty")
        d = step.to_dict()
        assert d["operation_class"] is None


# ===========================================================================
# Section 3 — Pipeline construction and queries
# ===========================================================================

class TestPipelineConstruction:

    def test_empty_pipeline(self):
        p = Pipeline(name="Empty")
        assert p.name == "Empty"
        assert len(p) == 0
        assert p.get_nodes() == []
        assert p.get_edges() == []
        assert p.get_sources() == []
        assert p.get_sinks() == []

    def test_add_step_increases_size(self):
        p = Pipeline()
        s1 = PipelineStep(name="A")
        s2 = PipelineStep(name="B")
        p.add_step(s1)
        assert len(p) == 1
        p.add_step(s2)
        assert len(p) == 2

    def test_add_duplicate_step_raises(self):
        p = Pipeline()
        step = PipelineStep(step_id="dup")
        p.add_step(step)
        dup = PipelineStep(step_id="dup", name="AlsoNamedDup")
        with pytest.raises(ValueError, match="already exists"):
            p.add_step(dup)
        assert len(p) == 1

    def test_get_step_returns_correct_step(self):
        p = Pipeline()
        step = PipelineStep(name="Target")
        p.add_step(step)
        retrieved = p.get_step(step.id)
        assert retrieved == step
        assert retrieved.name == "Target"

    def test_get_step_missing_raises_key_error(self):
        p = Pipeline()
        with pytest.raises(KeyError, match="not found"):
            p.get_step("nonexistent-id")

    def test_sources_and_sinks_linear(self, linear_pipeline, scaler_step, classifier_step):
        sources = linear_pipeline.get_sources()
        sinks = linear_pipeline.get_sinks()
        assert sources == [scaler_step]
        assert sinks == [classifier_step]

    def test_sources_and_sinks_branching(self):
        """
        Structure:
          Input → Scaler → Model1 ─┐
                         Model2 ─┤→ Ensemble
        """
        p = Pipeline()
        s_in = PipelineStep(name="Input", input_type=CleanTabular, output_type=CleanTabular)
        s_sc = PipelineStep(name="Scaler", input_type=CleanTabular, output_type=ScaledData)
        s_m1 = PipelineStep(name="Model1", input_type=ScaledData, output_type=ClassLabels)
        s_m2 = PipelineStep(name="Model2", input_type=ScaledData, output_type=ClassLabels)
        s_en = PipelineStep(name="Ensemble", input_type=ClassLabels, output_type=ClassLabels)

        for step in [s_in, s_sc, s_m1, s_m2, s_en]:
            p.add_step(step)

        p.add_edge(s_in.id, s_sc.id)
        p.add_edge(s_sc.id, s_m1.id)
        p.add_edge(s_sc.id, s_m2.id)   # fan-out
        p.add_edge(s_m1.id, s_en.id)
        p.add_edge(s_m2.id, s_en.id)   # fan-in

        assert set(p.get_sources()) == {s_in}
        assert set(p.get_sinks()) == {s_en}
        assert len(p.get_edges()) == 5
        assert len(p) == 5

    def test_topological_order_respects_dependencies(self):
        p = Pipeline()
        a = PipelineStep(name="A", output_type=CleanTabular)
        b = PipelineStep(name="B", input_type=CleanTabular, output_type=CleanTabular)
        c = PipelineStep(name="C", input_type=CleanTabular, output_type=CleanTabular)
        for s in [a, b, c]:
            p.add_step(s)
        p.add_edge(a.id, b.id)
        p.add_edge(b.id, c.id)

        order = p.topological_order()
        assert order.index(a) < order.index(b)
        assert order.index(b) < order.index(c)

    def test_predecessors_and_successors(self, linear_pipeline, scaler_step, classifier_step):
        assert linear_pipeline.predecessors(classifier_step) == [scaler_step]
        assert linear_pipeline.successors(scaler_step) == [classifier_step]
        assert linear_pipeline.predecessors(scaler_step) == []
        assert linear_pipeline.successors(classifier_step) == []


# ===========================================================================
# Section 4 — Edge validation
# ===========================================================================

class TestEdgeValidation:

    def test_add_edge_missing_source_raises(self):
        p = Pipeline()
        s = PipelineStep(name="S")
        p.add_step(s)
        with pytest.raises(ValueError, match="Source step"):
            p.add_edge("ghost", s.id)

    def test_add_edge_missing_destination_raises(self):
        p = Pipeline()
        s = PipelineStep(name="S")
        p.add_step(s)
        with pytest.raises(ValueError, match="Destination step"):
            p.add_edge(s.id, "ghost")

    def test_add_edge_type_mismatch_raises(self):
        """ClassLabels (output) → CleanTabular (input) is not compatible."""
        p = Pipeline()
        estimator = PipelineStep(name="Est", output_type=ClassLabels)
        scaler = PipelineStep(
            name="Scaler", input_type=CleanTabular, output_type=ScaledData
        )
        p.add_step(estimator)
        p.add_step(scaler)
        with pytest.raises(TypeError, match="Type mismatch"):
            p.add_edge(estimator.id, scaler.id)
        # Ensure the bad edge was NOT added
        assert len(p.get_edges()) == 0

    def test_add_edge_cycle_raises_and_rolls_back(self):
        p = Pipeline()
        a = PipelineStep(name="A")
        b = PipelineStep(name="B")
        c = PipelineStep(name="C")
        for s in [a, b, c]:
            p.add_step(s)
        p.add_edge(a.id, b.id)
        p.add_edge(b.id, c.id)

        with pytest.raises(ValueError, match="cycle"):
            p.add_edge(c.id, a.id)

        assert (c.id, a.id) not in p.get_edges()
        assert len(p.get_edges()) == 2   # unchanged

    def test_self_loop_raises(self):
        p = Pipeline()
        s = PipelineStep(name="Solo")
        p.add_step(s)
        with pytest.raises(ValueError, match="cycle"):
            p.add_edge(s.id, s.id)

    def test_compatible_types_do_not_raise(self):
        """ScaledData → ScaledData should be accepted (same type)."""
        p = Pipeline()
        a = PipelineStep(name="A", output_type=ScaledData)
        b = PipelineStep(name="B", input_type=ScaledData, output_type=ScaledData)
        p.add_step(a)
        p.add_step(b)
        p.add_edge(a.id, b.id)   # must not raise
        assert len(p.get_edges()) == 1


# ===========================================================================
# Section 5 — Clone and serialization
# ===========================================================================

class TestPipelineCloneAndSerialization:

    def test_clone_preserves_topology(self, linear_pipeline):
        cloned = linear_pipeline.clone()
        assert len(cloned) == len(linear_pipeline)
        assert len(cloned.get_edges()) == len(linear_pipeline.get_edges())

    def test_clone_has_new_step_ids(self, linear_pipeline):
        orig_ids = {s.id for s in linear_pipeline.get_nodes()}
        cloned_ids = {s.id for s in linear_pipeline.clone().get_nodes()}
        assert orig_ids.isdisjoint(cloned_ids)

    def test_clone_is_independent(self, linear_pipeline):
        cloned = linear_pipeline.clone()
        extra = PipelineStep(name="Extra")
        cloned.add_step(extra)
        assert len(linear_pipeline) == 2     # original unchanged
        assert len(cloned) == 3

    def test_to_dict_contains_required_keys(self, linear_pipeline):
        d = linear_pipeline.to_dict()
        assert "name" in d
        assert "steps" in d
        assert "edges" in d
        assert len(d["steps"]) == 2
        assert len(d["edges"]) == 1

    def test_structure_hash_is_deterministic(self, linear_pipeline):
        h1 = linear_pipeline.structure_hash()
        h2 = linear_pipeline.structure_hash()
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 16

    def test_structure_hash_differs_for_different_pipelines(self, linear_pipeline):
        other = Pipeline()
        a = PipelineStep(name="A", step_type="scaler",
                         operation=MinMaxScaler(),
                         input_type=CleanTabular, output_type=ScaledData)
        b = PipelineStep(name="B", step_type="classifier",
                         operation=LogisticRegression(),
                         input_type=ScaledData, output_type=ClassLabels)
        other.add_step(a)
        other.add_step(b)
        other.add_edge(a.id, b.id)
        # Different operation classes → different hash (MinMaxScaler vs StandardScaler)
        assert linear_pipeline.structure_hash() != other.structure_hash()


# ===========================================================================
# Section 6 — Pipeline execution on real data
# ===========================================================================

class TestPipelineExecution:

    def test_fit_and_predict_linear_pipeline(self, iris):
        X, y = iris
        scaler = PipelineStep(
            name="Scaler", step_type="scaler",
            operation=StandardScaler(),
            input_type=CleanTabular, output_type=ScaledData,
        )
        clf = PipelineStep(
            name="LR", step_type="classifier",
            operation=LogisticRegression(max_iter=200),
            input_type=ScaledData, output_type=ClassLabels,
        )
        p = Pipeline()
        p.add_step(scaler)
        p.add_step(clf)
        p.add_edge(scaler.id, clf.id)

        p.fit(X, y)
        preds = p.predict(X)

        assert len(preds) == len(y)
        accuracy = (preds == y).mean()
        assert accuracy > 0.8, f"Expected >80% accuracy on Iris, got {accuracy:.2%}"

    def test_predict_before_fit_raises(self, linear_pipeline, iris):
        X, _ = iris
        with pytest.raises(RuntimeError, match="not been fitted"):
            linear_pipeline.predict(X)

    def test_repr_and_str(self, linear_pipeline):
        r = repr(linear_pipeline)
        assert "LinearPipeline" in r
        assert "steps=2" in r
        s = str(linear_pipeline)
        assert "Scaler" in s
        assert "Classifier" in s


# ===========================================================================
# Section 7 — OperationRegistry
# ===========================================================================

class TestOperationRegistry:

    def test_default_registry_has_expected_types(self):
        types = default_registry.all_step_types()
        for expected in ["imputer", "scaler", "classifier", "regressor"]:
            assert expected in types, f"Missing step_type: {expected}"

    def test_get_by_type_returns_specs(self):
        specs = default_registry.get_by_type("scaler")
        assert len(specs) >= 2
        for spec in specs:
            assert spec.step_type == "scaler"

    def test_get_by_unknown_type_returns_empty(self):
        assert default_registry.get_by_type("flying_saucer") == []

    def test_sample_step_returns_valid_pipeline_step(self):
        step = default_registry.sample_step("scaler")
        assert isinstance(step, PipelineStep)
        assert step.step_type == "scaler"
        assert step.operation is not None
        assert step.input_type is not None
        assert step.output_type is not None

    def test_sample_step_unknown_type_raises(self):
        with pytest.raises(ValueError, match="No operations registered"):
            default_registry.sample_step("nonexistent_type")

    def test_sample_step_produces_different_results(self):
        """Sampling should not always return the same result."""
        results = {default_registry.sample_step("classifier").name for _ in range(20)}
        # With 6 classifiers registered, we expect at least 2 distinct ones in 20 draws
        assert len(results) >= 2

    def test_int_range_within_bounds(self):
        r = IntRange(5, 10)
        for _ in range(50):
            v = r.sample()
            assert 5 <= v <= 10

    def test_float_range_within_bounds(self):
        r = FloatRange(0.1, 1.0)
        for _ in range(50):
            v = r.sample()
            assert 0.1 <= v <= 1.0

    def test_float_range_log_scale(self):
        r = FloatRange(1e-4, 1.0, log_scale=True)
        values = [r.sample() for _ in range(100)]
        # Log scale should produce values spread across orders of magnitude
        assert min(values) < 0.01    # should occasionally be very small
        assert max(values) > 0.1     # should occasionally be large

    def test_categorical_samples_from_choices(self):
        c = Categorical(["a", "b", "c"])
        for _ in range(30):
            assert c.sample() in ["a", "b", "c"]

    def test_boolean_samples_both_values(self):
        b = Boolean()
        results = {b.sample() for _ in range(50)}
        assert True in results
        assert False in results

    def test_register_custom_spec(self):
        registry = OperationRegistry()
        custom_spec = OperationSpec(
            op_class=MinMaxScaler,
            step_type="custom_scaler",
            input_type=CleanTabular,
            output_type=ScaledData,
            search_space={},
        )
        registry.register(custom_spec)
        assert "custom_scaler" in registry.all_step_types()
        step = registry.sample_step("custom_scaler")
        assert isinstance(step.operation, MinMaxScaler)

    def test_sample_compatible_step(self):
        step = default_registry.sample_compatible_step(
            input_type=ScaledData,
            output_type=ClassLabels,
        )
        assert step is not None
        from c60.core.types import is_compatible as compat
        assert compat(ScaledData, step.input_type)
        assert compat(step.output_type, ClassLabels)

    def test_summary_is_printable(self):
        s = default_registry.summary()
        assert isinstance(s, str)
        assert "OperationRegistry" in s
        assert "scaler" in s
