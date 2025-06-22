"""
Tests for the graph_validator module.

This module contains unit tests for the graph validation logic.
"""

import pytest
from c60.engine.graph_schema import PipelineGraph
from c60.engine.graph_validator import GraphValidator


def test_validate_graph_structure():
    """Test basic graph structure validation."""
    # Create a valid graph
    graph = PipelineGraph()
    graph.add_node("preprocess", "preprocessor")
    graph.add_node("model", "estimator")
    graph.add_edge("preprocess", "model")
    
    # Should not raise any exceptions
    validator = GraphValidator()
    validator.validate(graph)


def test_detect_cycles():
    """Test cycle detection in the graph."""
    # Create a graph with a cycle
    graph = PipelineGraph()
    graph.add_node("node1", "transformer")
    graph.add_node("node2", "transformer")
    graph.add_node("node3", "transformer")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", "node3")
    graph.add_edge("node3", "node1")  # Creates a cycle
    
    validator = GraphValidator()
    with pytest.raises(ValueError, match="Cycle detected"):
        validator.validate(graph)


def test_validate_node_types():
    """Test validation of node types."""
    # Create graph with invalid node type
    graph = PipelineGraph()
    graph.add_node("invalid_node", "invalid_type")
    
    validator = GraphValidator()
    with pytest.raises(ValueError, match="Invalid node type"):
        validator.validate(graph)


def test_validate_required_parameters():
    """Test validation of required parameters."""
    # Create graph with missing required parameters
    graph = PipelineGraph()
    graph.add_node("model", "estimator", {"learning_rate": 0.01})  # Missing required 'n_estimators'
    
    validator = GraphValidator()
    with pytest.raises(ValueError, match="Missing required parameter"):
        validator.validate(graph)


def test_validate_edge_constraints():
    """Test validation of edge constraints."""
    # Create graph with invalid edge
    graph = PipelineGraph()
    graph.add_node("preprocess", "preprocessor")
    graph.add_node("model1", "estimator")
    graph.add_node("model2", "estimator")
    graph.add_edge("model1", "model2")  # Invalid: estimator output to another estimator
    
    validator = GraphValidator()
    with pytest.raises(ValueError, match="Invalid edge"):
        validator.validate(graph)
