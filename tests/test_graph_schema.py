"""
Tests for the graph_schema module.

This module contains unit tests for the DAG-based pipeline graph schema.
"""

import pytest
from c60.engine.graph_schema import Node, Edge, DAG


def test_node_creation():
    """Test creating a node with valid parameters."""
    node = Node(
        node_id="test_node",
        node_type="transformer",
        parameters={"param1": 42},
        description="Test node"
    )
    
    assert node.node_id == "test_node"
    assert node.node_type == "transformer"
    assert node.parameters == {"param1": 42}
    assert node.description == "Test node"


def test_edge_creation():
    """Test creating an edge between two nodes."""
    source = "node1"
    target = "node2"
    edge = Edge(source=source, target=target)
    
    assert edge.source == source
    assert edge.target == target


def test_dag_operations():
    """Test basic DAG operations."""
    dag = DAG()
    
    # Test adding nodes
    node1 = Node("node1", "transformer")
    node2 = Node("node2", "estimator")
    dag.add_node(node1)
    dag.add_node(node2)
    
    # Test adding edge
    edge = Edge("node1", "node2")
    dag.add_edge(edge)
    
    # Test getting nodes and edges
    assert len(dag.nodes) == 2
    assert len(dag.edges) == 1
    assert dag.get_node("node1") == node1
    assert dag.get_node("node2") == node2
    
    # Test topological sort
    topo_order = dag.topological_sort()
    assert len(topo_order) == 2
    assert topo_order[0] == "node1" or topo_order[1] == "node1"
    assert topo_order[0] == "node2" or topo_order[1] == "node2"


def test_dag_validation():
    """Test DAG validation for cycles."""
    dag = DAG()
    
    # Create a cycle: node1 -> node2 -> node3 -> node1
    dag.add_node(Node("node1", "transformer"))
    dag.add_node(Node("node2", "transformer"))
    dag.add_node(Node("node3", "transformer"))
    
    dag.add_edge(Edge("node1", "node2"))
    dag.add_edge(Edge("node2", "node3"))
    
    # Should not raise for DAG
    dag.validate()
    
    # Add cycle
    dag.add_edge(Edge("node3", "node1"))
    
    # Should raise for cycle
    with pytest.raises(ValueError, match="Cycle detected"):
        dag.validate()


def test_serialization():
    """Test serialization and deserialization of DAG."""
    # Create a simple DAG
    dag = DAG()
    dag.add_node(Node("node1", "transformer", {"param": 42}))
    dag.add_node(Node("node2", "estimator"))
    dag.add_edge(Edge("node1", "node2"))
    
    # Serialize
    serialized = dag.to_dict()
    
    # Deserialize
    new_dag = DAG.from_dict(serialized)
    
    # Verify
    assert len(new_dag.nodes) == 2
    assert len(new_dag.edges) == 1
    assert new_dag.get_node("node1").parameters == {"param": 42}
