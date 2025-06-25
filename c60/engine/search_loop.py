"""
Evolutionary search loop for optimizing ML pipelines using molecular GNNs.

Implements the core evolutionary search algorithm that uses a GNN to guide
search for optimal machine learning pipelines in C60.ai. Pipelines are treated as
molecules that can be mutated and recombined, inspired by molecular evolution.
"""
from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from .gnn_predictor import MolecularGNN, MolecularGraphEncoder


class PipelineMutator:
    """Handles mutation operations on pipeline graphs."""
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize the mutator.
        
        Args:
            mutation_rate: Probability of mutating each component
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, pipeline_graph: Dict) -> Dict:
        """
        Apply mutations to a pipeline graph.
        
        Args:
            pipeline_graph: The graph to mutate
            
        Returns:
            A new, mutated graph
        """
        # Create a copy of the graph
        mutated = {
            'x': pipeline_graph['x'].clone(),
            'edge_index': pipeline_graph['edge_index'].clone(),
            'edge_attr': pipeline_graph['edge_attr'].clone()
        }
        
        # Apply mutations to nodes
        for i in range(mutated['x'].size(0)):
            if random.random() < self.mutation_rate:
                # Simple mutation: add noise to node features
                noise = torch.randn_like(mutated['x'][i]) * 0.1
                mutated['x'][i] += noise
        
        # Apply mutations to edges
        for i in range(mutated['edge_attr'].size(0)):
            if random.random() < self.mutation_rate:
                # Simple mutation: add noise to edge features
                noise = torch.randn_like(mutated['edge_attr'][i]) * 0.1
                mutated['edge_attr'][i] += noise
        
        return mutated


class EvolutionarySearch:
    """
    Evolutionary search for optimizing ML pipelines using a GNN.
    
    This class implements a simple genetic algorithm where:
    1. A population of pipeline graphs is maintained
    2. Each graph is scored using a GNN predictor
    3. The best graphs are selected for reproduction
    4. New graphs are created through mutation
    5. The process repeats for a number of generations
    """
    
    def __init__(
        self,
        gnn: MolecularGNN,
        encoder: MolecularGraphEncoder,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        elite_frac: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize the evolutionary search.
        
        Args:
            gnn: Pre-trained GNN for pipeline scoring
            encoder: Encoder for converting pipelines to graphs
            population_size: Number of pipelines in each generation
            n_generations: Number of generations to run
            mutation_rate: Probability of mutating each component
            elite_frac: Fraction of top performers to keep in each generation
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.gnn = gnn.to(device)
        self.encoder = encoder
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac
        self.device = device
        self.mutator = PipelineMutator(mutation_rate)
        
        # Store population and scores
        self.population: List[Dict] = []
        self.scores: List[float] = []
    
    def initialize_population(self, initial_pipeline) -> None:
        """
        Initialize the population with random variations of the initial pipeline.
        
        Args:
            initial_pipeline: The starting pipeline
        """
        self.population = []
        
        # Encode the initial pipeline
        base_graph = self.encoder.encode_pipeline(initial_pipeline)
        
        # Create population by mutating the base graph
        for _ in range(self.population_size):
            mutated = self.mutator.mutate(base_graph)
            self.population.append(mutated)
    
    def evaluate_population(self, X, y) -> None:
        """
        Evaluate all pipelines in the current population.
        
        Args:
            X: Input features
            y: Target values
        """
        self.scores = []
        
        # Convert graphs to PyG Data objects
        data_list = []
        for graph in self.population:
            data = Data(
                x=graph['x'],
                edge_index=graph['edge_index'],
                edge_attr=graph['edge_attr'],
                batch=torch.zeros(graph['x'].size(0), dtype=torch.long)
            )
            data_list.append(data)
        
        # Create batch and move to device
        batch = Batch.from_data_list(data_list).to(self.device)
        
        # Get predictions from GNN
        with torch.no_grad():
            self.gnn.eval()
            scores = self.gnn(batch).squeeze().cpu().numpy()
        
        # Store scores (higher is better)
        self.scores = scores.tolist()
    
    def select_parents(self) -> List[Dict]:
        """
        Select parents for the next generation using tournament selection.
        
        Returns:
            List of selected parent graphs
        """
        tournament_size = 3
        parents = []
        
        while len(parents) < self.population_size:
            # Randomly select tournament participants
            participants = random.sample(
                list(zip(self.population, self.scores)),
                min(tournament_size, len(self.population))
            )
            
            # Select the best participant
            best = max(participants, key=lambda x: x[1])
            parents.append(best[0])
        
        return parents
    
    def create_next_generation(self, parents: List[Dict]) -> None:
        """
        Create the next generation from the selected parents.
        
        Args:
            parents: List of parent graphs
        """
        # Keep elite solutions
        n_elite = int(self.population_size * self.elite_frac)
        elite_indices = np.argsort(self.scores)[-n_elite:]
        elite = [self.population[i] for i in elite_indices]
        
        # Create new population with mutations of parents
        new_population = []
        
        # Add elite solutions
        new_population.extend(elite)
        
        # Add mutated parents
        while len(new_population) < self.population_size:
            parent = random.choice(parents)
            child = self.mutator.mutate(parent)
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
    
    def search(self, initial_pipeline, X, y) -> Tuple[Dict, float]:
        """
        Run the evolutionary search.
        
        Args:
            initial_pipeline: The starting pipeline
            X: Input features
            y: Target values
            
        Returns:
            Tuple of (best_pipeline, best_score)
        """
        # Initialize population
        self.initialize_population(initial_pipeline)
        
        best_score = -float('inf')
        best_pipeline = None
        
        # Evolutionary loop
        for gen in range(self.n_generations):
            # Evaluate current population
            self.evaluate_population(X, y)
            
            # Track best solution
            gen_best_idx = np.argmax(self.scores)
            if self.scores[gen_best_idx] > best_score:
                best_score = self.scores[gen_best_idx]
                best_pipeline = self.population[gen_best_idx]
            
            # Log progress
            print(f"Generation {gen+1}/{self.n_generations} - "
                  f"Best: {best_score:.4f}, "
                  f"Avg: {np.mean(self.scores):.4f}")
            
            # Select parents
            parents = self.select_parents()
            
            # Create next generation
            self.create_next_generation(parents)
        
        return best_pipeline, best_score
