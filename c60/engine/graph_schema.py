"""
Graph schema utilities for the C60.ai framework.

Defines classes and functions for defining and validating graph schemas
used in graph-based machine learning pipelines in C60.ai.
"""

import networkx as nx
from typing import Dict, Any, List, Optional

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypeVar
from enum import Enum
import json
import networkx as nx

# Type variable for generic type hints
T = TypeVar('T')

class NodeType(str, Enum):
    """Types of nodes in the pipeline DAG."""
    TRANSFORMER = "transformer"
    ESTIMATOR = "estimator"
    PREPROCESSOR = "preprocessor"
    FEATURE_SELECTOR = "feature_selector"
    IMBALANCE_HANDLER = "imbalance_handler"

@dataclass
class Node:
    """A node in the pipeline DAG.
    
    Attributes:
        node_id: Unique identifier for the node
        node_type: Type of the node (transformer, estimator, etc.)
        parameters: Dictionary of parameters for the node
        description: Optional description of the node
    """
    node_id: str
    node_type: NodeType
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Edge:
    """A directed edge between two nodes in the pipeline DAG.
    
    Attributes:
        source: ID of the source node
        target: ID of the target node
    """
    source: str
    target: str


class DAG:
    """
    A Directed Acyclic Graph (DAG) representing a machine learning pipeline.
    
    This class provides methods to manipulate and query the pipeline graph,
    including adding/removing nodes and edges, validating the graph structure,
    and serializing/deserializing the graph.
    """
    
    def __init__(self):
        """Initialize an empty DAG."""
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._graph: Optional[nx.DiGraph] = None
    
    def add_node(self, node: Node) -> None:
        """Add a node to the DAG.
        
        Args:
            node: The node to add
            
        Raises:
            ValueError: If a node with the same ID already exists
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID {node.node_id} already exists")
        self.nodes[node.node_id] = node
        self._graph = None  # Invalidate cached graph
    
    def add_edge(self, edge: Edge) -> None:
        """Add a directed edge to the DAG.
        
        Args:
            edge: The edge to add
            
        Raises:
            ValueError: If the edge would create a cycle or nodes don't exist
        """
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the graph")
        
        # Check for cycles
        temp_edges = self.edges + [edge]
        if self._has_cycle(temp_edges):
            raise ValueError(f"Adding edge {edge.source} -> {edge.target} would create a cycle")
        
        self.edges.append(edge)
        self._graph = None  # Invalidate cached graph
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(node_id)
    
    def topological_sort(self) -> List[str]:
        """Perform a topological sort of the DAG.
        
        Returns:
            A list of node IDs in topological order
            
        Raises:
            ValueError: If the graph contains cycles
        """
        self._build_graph()
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph contains at least one cycle")
    
    def validate(self) -> None:
        """Validate the DAG structure.
        
        Raises:
            ValueError: If the graph is invalid (contains cycles, etc.)
        """
        self.topological_sort()  # Will raise if cycles exist
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the DAG to a dictionary.
        
        Returns:
            A dictionary representation of the DAG
        """
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type.value,
                    "parameters": n.parameters,
                    "description": n.description
                }
                for n in self.nodes.values()
            ],
            "edges": [{"source": e.source, "target": e.target} for e in self.edges]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DAG':
        """Deserialize a DAG from a dictionary.
        
        Args:
            data: Dictionary containing DAG data
            
        Returns:
            A new DAG instance
        """
        dag = cls()
        
        # Add nodes
        for node_data in data["nodes"]:
            node = Node(
                node_id=node_data["node_id"],
                node_type=NodeType(node_data["node_type"]),
                parameters=node_data.get("parameters", {}),
                description=node_data.get("description", "")
            )
            dag.add_node(node)
        
        # Add edges
        for edge_data in data["edges"]:
            edge = Edge(
                source=edge_data["source"],
                target=edge_data["target"]
            )
            dag.add_edge(edge)
        
        return dag
    
    def _build_graph(self) -> None:
        """Build the NetworkX graph representation if not already built."""
        if self._graph is None:
            self._graph = nx.DiGraph()
            self._graph.add_nodes_from(self.nodes.keys())
            self._graph.add_edges_from([(e.source, e.target) for e in self.edges])
    
    def _has_cycle(self, edges: List[Edge]) -> bool:
        """Check if adding an edge would create a cycle.
        
        Args:
            edges: List of edges to check
            
        Returns:
            True if adding the edge would create a cycle, False otherwise
            
        Raises:
            ImportError: If networkx is not installed
        """
        if not hasattr(nx, 'DiGraph'):
            raise ImportError("networkx is required for cycle detection")
            
        temp_graph = nx.DiGraph()
        temp_graph.add_nodes_from(self.nodes.keys())
        temp_graph.add_edges_from([(e.source, e.target) for e in edges])
        
        try:
            list(nx.topological_sort(temp_graph))
            return False
        except nx.NetworkXUnfeasible:
            return True


# For backward compatibility with existing code
PipelineGraph = DAG