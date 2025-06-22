"""Tests for the evolutionary search implementation."""
import unittest
import torch
import numpy as np
from unittest.mock import MagicMock

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.search_loop import PipelineMutator, EvolutionarySearch
from engine.gnn_predictor import MolecularGNN, MolecularGraphEncoder


class TestPipelineMutator(unittest.TestCase):
    """Test cases for the PipelineMutator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mutator = PipelineMutator(mutation_rate=0.5)  # High rate for testing
        
        # Create a simple test graph
        self.test_graph = {
            'x': torch.randn(3, 64),  # 3 nodes, 64 features each
            'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            'edge_attr': torch.randn(2, 16)  # 2 edges, 16 features each
        }
    
    def test_mutation_changes_graph(self):
        """Test that mutation changes the graph."""
        # Make multiple attempts to ensure mutation happens (due to random chance)
        for _ in range(10):
            mutated = self.mutator.mutate(self.test_graph)
            
            # Check that at least one node or edge was modified
            node_changed = not torch.allclose(mutated['x'], self.test_graph['x'])
            edge_changed = not torch.allclose(mutated['edge_attr'], self.test_graph['edge_attr'])
            
            if node_changed or edge_changed:
                break
        
        self.assertTrue(node_changed or edge_changed, "No mutations occurred")
    
    def test_mutation_preserves_shape(self):
        """Test that mutation preserves graph structure."""
        mutated = self.mutator.mutate(self.test_graph)
        
        # Check shapes are preserved
        self.assertEqual(mutated['x'].shape, self.test_graph['x'].shape)
        self.assertEqual(mutated['edge_index'].shape, self.test_graph['edge_index'].shape)
        self.assertEqual(mutated['edge_attr'].shape, self.test_graph['edge_attr'].shape)


class TestEvolutionarySearch(unittest.TestCase):
    """Test cases for the EvolutionarySearch class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock GNN and encoder
        self.mock_gnn = MagicMock(spec=MolecularGNN)
        self.mock_encoder = MagicMock(spec=MolecularGraphEncoder)
        
        # Set up mock return values
        self.mock_encoder.encode_pipeline.return_value = {
            'x': torch.randn(3, 64),
            'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            'edge_attr': torch.randn(2, 16)
        }
        
        # Mock GNN to return random scores
        self.mock_gnn.return_value = torch.randn(1, 1)
        
        # Create search instance
        self.search = EvolutionarySearch(
            gnn=self.mock_gnn,
            encoder=self.mock_encoder,
            population_size=10,
            n_generations=5,
            mutation_rate=0.5
        )
    
    def test_initialization(self):
        """Test that search initializes correctly."""
        self.assertEqual(len(self.search.population), 0)
        self.assertEqual(len(self.search.scores), 0)
    
    def test_initialize_population(self):
        """Test population initialization."""
        mock_pipeline = MagicMock()
        self.search.initialize_population(mock_pipeline)
        
        # Check population size is correct
        self.assertEqual(len(self.search.population), self.search.population_size)
        
        # Check all graphs have the expected structure
        for graph in self.search.population:
            self.assertIn('x', graph)
            self.assertIn('edge_index', graph)
            self.assertIn('edge_attr', graph)
    
    def test_evaluate_population(self):
        """Test population evaluation."""
        # Set up test data
        X = torch.randn(10, 5)  # 10 samples, 5 features
        y = torch.randn(10)     # 10 targets
        
        # Mock GNN to return predictable scores
        self.mock_gnn.return_value = torch.ones(1, 1) * 0.5
        
        # Initialize and evaluate population
        mock_pipeline = MagicMock()
        self.search.initialize_population(mock_pipeline)
        self.search.evaluate_population(X, y)
        
        # Check scores were set correctly
        self.assertEqual(len(self.search.scores), self.search.population_size)
        self.assertTrue(all(score == 0.5 for score in self.search.scores))
    
    def test_select_parents(self):
        """Test parent selection."""
        # Set up test population with known scores
        self.search.population = [{} for _ in range(10)]
        self.search.scores = [i for i in range(10)]  # 0 to 9
        
        # Select parents
        parents = self.search.select_parents()
        
        # Check we got the right number of parents
        self.assertEqual(len(parents), self.search.population_size)
    
    def test_create_next_generation(self):
        """Test creation of next generation."""
        # Set up test population
        self.search.population = [{'score': i} for i in range(10)]
        self.search.scores = list(range(10))  # 0 to 9
        
        # Create next generation
        parents = [{'score': 5}] * 10  # All parents have score 5
        self.search.create_next_generation(parents)
        
        # Check population size is preserved
        self.assertEqual(len(self.search.population), self.search.population_size)
        
        # Check elite solutions were preserved
        self.assertIn({'score': 9}, self.search.population)  # Best solution
        self.assertIn({'score': 8}, self.search.population)  # Second best solution


if __name__ == '__main__':
    unittest.main()
