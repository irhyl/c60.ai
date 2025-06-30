"""
Molecular Graph Neural Network for pipeline performance prediction.

GNN predictor module for the C60.ai framework.

Defines the MolecularGNN and supporting classes for graph neural network-based prediction
in machine learning pipelines in C60.ai.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from typing import Optional, Any, Dict


class MolecularGNN(nn.Module):
    """
    Graph Neural Network for processing molecular graph representations of ML pipelines.
    
    This model takes a molecular graph representation of an ML pipeline and predicts
    its performance metrics. The molecular graph encodes pipeline components as atoms
    and their relationships as bonds.
    
    Args:
        node_feature_dim: Dimensionality of node features
        edge_feature_dim: Dimensionality of edge features
        hidden_dim: Hidden dimension size
        num_layers: Number of GNN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        node_feature_dim: int = 64,
        edge_feature_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node feature transformation
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Edge feature transformation
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GCNConv(hidden_dim, hidden_dim)
            )
        
        # Prediction head
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Predict single performance metric
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_feature_dim]
                - batch: Batch vector [num_nodes]
                
        Returns:
            Prediction tensor [batch_size, 1]
        """
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch
        
        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling and prediction
        x = self.pool(x, batch)
        return self.classifier(x)


# --- Clean code: Move component/edge type mappings to module-level constants ---
# If these lists grow, consider moving them to a config or registry module.
COMPONENT_TYPES = {
    # Data preprocessing
    'imputer': 0,
    'scaler': 1,
    'encoder': 2,
    'feature_selector': 3,
    # Feature engineering
    'pca': 10,
    'kernel_pca': 11,
    'feature_aggregation': 12,
    # Models
    'classifier': 20,
    'regressor': 21,
    'clusterer': 22,
    # Other
    'other': 99
}

EDGE_TYPES = {
    'data_flow': 0,
    'control_flow': 1,
    'feature_flow': 2
}

class MolecularGraphEncoder:
    """
    Encodes ML pipelines into molecular graph representations.
    
    This class handles the conversion of ML pipeline components into a molecular
    graph structure that can be processed by the MolecularGNN.
    
    Extension:
        To add new component or edge types, update the COMPONENT_TYPES and EDGE_TYPES constants at the module level.
        If these lists grow, consider moving them to a config or registry module.
    """
    
    def __init__(self):
        # Use module-level constants for type mappings
        self.component_types = COMPONENT_TYPES
        self.edge_types = EDGE_TYPES
    
    def encode_pipeline(self, pipeline) -> Dict:
        """
        Encode an ML pipeline into a molecular graph.
        
        Args:
            pipeline: The ML pipeline to encode
            
        Returns:
            Dictionary containing:
                - node_features: Tensor of node features
                - edge_index: Tensor of edge indices
                - edge_attrs: Tensor of edge attributes
        """
        # TODO: Implement actual pipeline to graph conversion
        # This is a placeholder implementation
        
        # Example: Create a simple graph with one node per pipeline step
        num_nodes = len(pipeline.steps) if hasattr(pipeline, 'steps') else 1
        
        # Node features: [component_type, num_params, is_trainable, ...]
        node_features = torch.zeros((num_nodes, 64))  # 64-dim node features
        
        # Simple edge connectivity - fully connected for now
        edge_index = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected edges
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Edge attributes
        edge_attrs = torch.ones((edge_index.size(1), 16))  # 16-dim edge features
        
        return {
            'x': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attrs
        }
