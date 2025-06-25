"""
Graph validator utilities for the C60.ai framework.

Defines functions for validating graph schemas and pipeline DAGs in C60.ai.
"""

import networkx as nx
from typing import Any, Dict, List, Optional, Set
from .graph_schema import Node, Edge, DAG


class GraphValidationError(Exception):
    """Exception raised for graph validation errors."""
    pass


class GraphValidator:
    """
    Validates pipeline graphs to ensure they are properly structured.
    
    This class provides methods to validate the structure of pipeline graphs,
    including checking for cycles, disconnected components, and invalid
    node/edge configurations.
    """
    
    def __init__(self):
        """Initialize the GraphValidator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, graph: DAG) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a pipeline graph.
        
        Args:
            graph: The DAG to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._check_nodes(graph)
        self._check_edges(graph)
        self._check_connectivity(graph)
        self._check_for_cycles(graph)
        self._check_node_types(graph)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _check_nodes(self, graph: DAG) -> None:
        """Check that all nodes are valid."""
        if not graph.nodes:
            self.errors.append("Graph has no nodes")
            return
        
        # Check for duplicate node IDs
        node_ids = [node.id for node in graph.nodes]
        if len(node_ids) != len(set(node_ids)):
            self.errors.append("Duplicate node IDs found")
        
        # Check for nodes with invalid types
        for node in graph.nodes:
            if not isinstance(node.id, str):
                self.errors.append(f"Node ID must be a string, got {type(node.id).__name__}")
            if not node.node_type:
                self.errors.append(f"Node {node.id} has no type")
    
    def _check_edges(self, graph: DAG) -> None:
        """Check that all edges are valid."""
        if not graph.edges:
            self.warnings.append("Graph has no edges")
            return
        
        node_ids = {node.id for node in graph.nodes}
        
        for edge in graph.edges:
            # Check if source and target nodes exist
            if edge.source not in node_ids:
                self.errors.append(f"Edge references non-existent source node: {edge.source}")
            if edge.target not in node_ids:
                self.errors.append(f"Edge references non-existent target node: {edge.target}")
            
            # Check for self-loops
            if edge.source == edge.target:
                self.errors.append(f"Self-loop detected on node: {edge.source}")
    
    def _check_connectivity(self, graph: DAG) -> None:
        """Check that the graph is fully connected."""
        if not graph.nodes or not graph.edges:
            return
        
        # Build adjacency list
        adj: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}
        for edge in graph.edges:
            adj[edge.source].append(edge.target)
        
        # Find all reachable nodes from the first node
        visited: Set[str] = set()
        stack = [graph.nodes[0].id] if graph.nodes else []
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(adj[node])
        
        # Check if all nodes are reachable
        unreachable = set(node.id for node in graph.nodes) - visited
        if unreachable:
            self.warnings.append(f"Graph has {len(unreachable)} unreachable nodes: {', '.join(sorted(unreachable))}")
    
    def _check_for_cycles(self, graph: DAG) -> None:
        """Check if the graph contains any cycles."""
        if not graph.nodes or not graph.edges:
            return
        
        # Build adjacency list
        adj: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}
        for edge in graph.edges:
            adj[edge.source].append(edge.target)
        
        # Use depth-first search to detect cycles
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph.nodes:
            if node.id not in visited:
                if has_cycle(node.id):
                    self.errors.append("Cycle detected in the graph")
                    break
    
    def _check_node_types(self, graph: DAG) -> None:
        """Check that node types are valid and compatible."""
        # This is a simplified example - in a real implementation, you would
        # check that node types are compatible with their inputs/outputs
        valid_types = {
            'data_loader', 'preprocessor', 'transformer',
            'feature_selector', 'model', 'evaluator'
        }
        
        for node in graph.nodes:
            if node.node_type not in valid_types:
                self.warnings.append(f"Node {node.id} has unknown type: {node.node_type}")


def validate_graph(graph: DAG) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a pipeline graph.
    
    This is a convenience function that creates a GraphValidator instance
    and calls validate() on it.
    
    Args:
        graph: The DAG to validate
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = GraphValidator()
    return validator.validate(graph)
