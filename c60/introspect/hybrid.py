"""
Hybrid node implementation for combining symbolic and neural components.
"""
from typing import Any, Optional


class HybridNode:
    """
    Represents a node in the pipeline that can be symbolic (rule-based) or neural (learned).
    """
    def __init__(self, node_id: str, node_type: str, symbolic_rule: Optional[Any] = None, neural_model: Optional[Any] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.symbolic_rule = symbolic_rule
        self.neural_model = neural_model

    def is_symbolic(self) -> bool:
        """Check if the node has a symbolic rule."""
        return self.symbolic_rule is not None

    def is_neural(self) -> bool:
        """Check if the node has a neural model."""
        return self.neural_model is not None

    def __repr__(self) -> str:
        if self.is_symbolic() and self.is_neural():
            return f"HybridNode({self.node_id}, symbolic+neural)"
        elif self.is_symbolic():
            return f"HybridNode({self.node_id}, symbolic)"
        elif self.is_neural():
            return f"HybridNode({self.node_id}, neural)"
        else:
            return f"HybridNode({self.node_id}, empty)"

    def predict(self, data: Any) -> Any:
        """
        Make a prediction using the node's model or rule.
        
        Args:
            data: Input data for prediction
            
        Returns:
            The prediction result
        """
        if self.is_neural():
            return self.neural_model.predict(data)
        elif self.is_symbolic():
            return self.symbolic_rule(data)
        raise ValueError("Node has no prediction capability (neither neural model nor symbolic rule)")
