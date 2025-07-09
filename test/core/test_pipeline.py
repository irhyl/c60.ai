import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
import uuid

from src.c60.core.pipeline import PipelineStep, Pipeline

# Conceptual Git Commit 2/4: Add tests for PipelineStep initialization and properties.
def test_pipeline_step_initialization():
    """
    Test that PipelineStep objects are initialized correctly with
    provided and default values.
    """
    step1 = PipelineStep(name="Scaler", step_type="transformer", operation=StandardScaler())
    assert step1.name == "Scaler"
    assert step1.step_type == "transformer"
    assert isinstance(step1.operation, StandardScaler)
    assert isinstance(step1.id, str)
    assert len(step1.id) > 0
    assert step1.params == {}

    step2 = PipelineStep(step_id="custom_id_123", name="Model", params={"C": 0.1})
    assert step2.id == "custom_id_123"
    assert step2.name == "Model"
    assert step2.step_type == "generic" # Default type
    assert step2.operation is None
    assert step2.params == {"C": 0.1}

def test_pipeline_step_equality_and_hashing():
    """
    Test that PipelineStep equality and hashing work based on their unique ID.
    This is important for graph operations (networkx uses node IDs).
    """
    step_id = str(uuid.uuid4())
    step1 = PipelineStep(step_id=step_id, name="StepA")
    step2 = PipelineStep(step_id=step_id, name="StepA") # Same ID
    step3 = PipelineStep(name="StepB") # Different ID

    assert step1 == step2
    assert step1 != step3
    assert hash(step1) == hash(step2)
    assert hash(step1) != hash(step3) # Highly probable, though not strictly guaranteed for different hashes
    assert {step1, step2} == {step1} # Set should treat them as the same

# Conceptual Git Commit 3/4: Add tests for Pipeline step and edge addition (valid cases).
def test_pipeline_initialization():
    """
    Test that a Pipeline object initializes with no steps or edges.
    """
    pipeline = Pipeline(name="Test Pipeline")
    assert pipeline.name == "Test Pipeline"
    assert len(pipeline) == 0
    assert pipeline.get_nodes() == []
    assert pipeline.get_edges() == []
    assert pipeline.get_sources() == []
    assert pipeline.get_sinks() == []

def test_pipeline_add_step():
    """
    Test adding individual steps to the pipeline.
    """
    pipeline = Pipeline()
    step1 = PipelineStep(name="Scaler")
    step2 = PipelineStep(name="Model")

    pipeline.add_step(step1)
    assert len(pipeline) == 1
    assert pipeline.get_nodes() == [step1]

    pipeline.add_step(step2)
    assert len(pipeline) == 2
    assert set(pipeline.get_nodes()) == {step1, step2} # Use set for order-independent comparison

def test_pipeline_add_edge_valid():
    """
    Test adding valid edges between steps in the pipeline.
    """
    pipeline = Pipeline()
    step_a = PipelineStep(name="A")
    step_b = PipelineStep(name="B")
    step_c = PipelineStep(name="C")

    pipeline.add_step(step_a)
    pipeline.add_step(step_b)
    pipeline.add_step(step_c)

    pipeline.add_edge(step_a.id, step_b.id)
    assert pipeline.get_edges() == [(step_a.id, step_b.id)]

    pipeline.add_edge(step_b.id, step_c.id)
    assert set(pipeline.get_edges()) == {(step_a.id, step_b.id), (step_b.id, step_c.id)}
    assert len(pipeline.get_edges()) == 2

    # Test sources and sinks for a linear pipeline
    assert pipeline.get_sources() == [step_a]
    assert pipeline.get_sinks() == [step_c]

def test_pipeline_complex_structure_sources_sinks():
    """
    Test sources and sinks for a more complex, branching pipeline structure.
    """
    pipeline = Pipeline()
    s_input = PipelineStep(name="Input")
    s_scaler = PipelineStep(name="Scaler")
    s_poly = PipelineStep(name="PolyFeatures")
    s_model1 = PipelineStep(name="Model1")
    s_model2 = PipelineStep(name="Model2")
    s_ensemble = PipelineStep(name="Ensembler")

    for step in [s_input, s_scaler, s_poly, s_model1, s_model2, s_ensemble]:
        pipeline.add_step(step)

    pipeline.add_edge(s_input.id, s_scaler.id)
    pipeline.add_edge(s_scaler.id, s_poly.id)
    pipeline.add_edge(s_poly.id, s_model1.id)
    pipeline.add_scaler = PipelineStep(name="Scaler")
    pipeline.add_edge(s_scaler.id, s_model2.id) # Branching
    pipeline.add_edge(s_model1.id, s_ensemble.id)
    pipeline.add_edge(s_model2.id, s_ensemble.id)

    assert set(pipeline.get_sources()) == {s_input}
    assert set(pipeline.get_sinks()) == {s_ensemble}
    assert len(pipeline.get_edges()) == 6
    assert len(pipeline) == 6

def test_pipeline_get_step():
    """
    Test retrieving a step by its ID.
    """
    pipeline = Pipeline()
    step1 = PipelineStep(name="Step1")
    pipeline.add_step(step1)
    retrieved_step = pipeline.get_step(step1.id)
    assert retrieved_step == step1
    assert retrieved_step.name == "Step1"

# Conceptual Git Commit 4/4: Add tests for Pipeline error handling (duplicate steps, cycles, non-existent steps/edges).
def test_pipeline_add_duplicate_step_raises_error():
    """
    Test that adding a step with an existing ID raises a ValueError.
    """
    pipeline = Pipeline()
    step1 = PipelineStep(step_id="common_id", name="First Step")
    step2 = PipelineStep(step_id="common_id", name="Second Step") # Same ID as step1

    pipeline.add_step(step1)
    with pytest.raises(ValueError, match="Step with ID 'common_id' already exists"):
        pipeline.add_step(step2)
    assert len(pipeline) == 1 # Ensure no duplicate was added

def test_pipeline_add_edge_non_existent_step_raises_error():
    """
    Test that adding an edge with non-existent step IDs raises a ValueError.
    """
    pipeline = Pipeline()
    step_a = PipelineStep(name="A")
    pipeline.add_step(step_a)

    with pytest.raises(ValueError, match="Source step with ID 'non_existent' not found"):
        pipeline.add_edge("non_existent", step_a.id)

    with pytest.raises(ValueError, match="Destination step with ID 'non_existent' not found"):
        pipeline.add_edge(step_a.id, "non_existent")

def test_pipeline_add_edge_creates_cycle_raises_error():
    """
    Test that adding an edge which would create a cycle raises a ValueError
    and the edge is not added. This is critical for DAG validation.
    """
    pipeline = Pipeline()
    step1 = PipelineStep(name="Step1")
    step2 = PipelineStep(name="Step2")
    step3 = PipelineStep(name="Step3")

    pipeline.add_step(step1)
    pipeline.add_step(step2)
    pipeline.add_step(step3)

    pipeline.add_edge(step1.id, step2.id)
    pipeline.add_edge(step2.id, step3.id)

    # Attempt to create a cycle: step3 -> step1
    with pytest.raises(ValueError, match="would create a cycle"):
        pipeline.add_edge(step3.id, step1.id)

    # Verify the edge was NOT added after the error
    assert (step3.id, step1.id) not in pipeline.get_edges()
    assert len(pipeline.get_edges()) == 2 # Should still only have the original two edges

    # Test a self-loop attempt
    with pytest.raises(ValueError, match="would create a cycle"):
        pipeline.add_edge(step1.id, step1.id)
    assert (step1.id, step1.id) not in pipeline.get_edges()

def test_pipeline_get_step_non_existent_raises_keyerror():
    """
    Test that retrieving a non-existent step raises a KeyError.
    """
    pipeline = Pipeline()
    with pytest.raises(KeyError, match="Step with ID 'non_existent' not found"):
        pipeline.get_step("non_existent")
