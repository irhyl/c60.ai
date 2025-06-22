"""Tests for the MolecularGNN implementation."""
import unittest
import torch
from torch_geometric.data import Data, Batch

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c60.gnn.predictor import MolecularGNN
from c60.gnn.pipeline_to_graph import MolecularGraphEncoder


class TestMolecularGNN(unittest.TestCase):
    """Test cases for the MolecularGNN class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small test graph
        self.node_feature_dim = 64
        self.edge_feature_dim = 16
        self.hidden_dim = 128
        
        # Test graph with 3 nodes and 2 edges
        self.test_graph = Data(
            x=torch.randn(3, self.node_feature_dim),  # 3 nodes
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            edge_attr=torch.randn(2, self.edge_feature_dim),  # 2 edges
            batch=torch.tensor([0, 0, 0])  # All nodes in same graph
        )
        
        # Create model
        self.model = MolecularGNN(
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        # Forward pass
        out = self.model(self.test_graph)
        
        # Check output shape
        self.assertEqual(out.shape, (1, 1))  # Batch size 1, output dim 1
    
    def test_batch_processing(self):
        """Test processing a batch of graphs."""
        # Create a batch of 2 graphs
        graph1 = Data(
            x=torch.randn(2, self.node_feature_dim),
            edge_index=torch.tensor([[0, 1]], dtype=torch.long).t(),
            edge_attr=torch.randn(1, self.edge_feature_dim),
            batch=torch.tensor([0, 0])
        )
        
        graph2 = Data(
            x=torch.randn(3, self.node_feature_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            edge_attr=torch.randn(2, self.edge_feature_dim),
            batch=torch.tensor([1, 1, 1])
        )
        
        # Create batch
        batch = Batch.from_data_list([graph1, graph2])
        
        # Forward pass
        out = self.model(batch)
        
        # Check output shape
        self.assertEqual(out.shape, (2, 1))  # Batch size 2, output dim 1


class TestMolecularGraphEncoder(unittest.TestCase):
    """Test cases for the MolecularGraphEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = MolecularGraphEncoder()
    
    def test_encode_pipeline(self):
        """Test encoding a pipeline."""
        # Create a mock pipeline object
        class MockPipeline:
            def __init__(self):
                self.steps = [
                    ('imputer', None),
                    ('scaler', None),
                    ('classifier', None)
                ]
        
        pipeline = MockPipeline()
        
        # Encode pipeline
        graph = self.encoder.encode_pipeline(pipeline)
        
        # Check output structure
        self.assertIn('x', graph)
        self.assertIn('edge_index', graph)
        self.assertIn('edge_attr', graph)
        
        # Check shapes
        self.assertEqual(graph['x'].shape[0], 3)  # 3 nodes
        self.assertEqual(graph['edge_index'].shape[0], 2)  # Edge index has 2 rows
        self.assertEqual(graph['edge_attr'].shape[0], graph['edge_index'].shape[1])  # One attr per edge


if __name__ == '__main__':
    unittest.main()
