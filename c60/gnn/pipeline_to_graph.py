"""Utility for converting pipeline objects to graph representations."""
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from ..core.pipeline import Pipeline, PipelineStep

class PipelineGraphConverter:
    """Converts Pipeline objects to graph representations for GNNs."""
    
    def __init__(self, 
                 node_feature_dim: int = 32,
                 max_steps: int = 20):
        """
        Initialize the pipeline graph converter.
        
        Args:
            node_feature_dim: Dimensionality of node features.
            max_steps: Maximum number of steps in a pipeline.
        """
        self.node_feature_dim = node_feature_dim
        self.max_steps = max_steps
        self.step_types = [
            'imputer', 'scaler', 'encoder', 'selector', 'classifier', 'regressor', 'other'
        ]
        self._init_step_type_embeddings()
    
    def _init_step_type_embeddings(self):
        """Initialize embeddings for different step types."""
        # Create a mapping from step type to embedding
        self.step_type_to_idx = {t: i for i, t in enumerate(self.step_types)}
        self.step_type_embeddings = torch.randn(
            len(self.step_types), 
            self.node_feature_dim
        )
    
    def _get_step_type_embedding(self, step: PipelineStep) -> torch.Tensor:
        """Get the embedding for a step type."""
        step_type = self._classify_step_type(step)
        idx = self.step_type_to_idx[step_type]
        return self.step_type_embeddings[idx]
    
    def _classify_step_type(self, step: PipelineStep) -> str:
        """Classify the type of a pipeline step."""
        step_name = step.name.lower()
        
        if 'imput' in step_name:
            return 'imputer'
        elif 'scale' in step_name or 'normaliz' in step_name:
            return 'scaler'
        elif 'encod' in step_name or 'categor' in step_name:
            return 'encoder'
        elif 'select' in step_name or 'pca' in step_name or 'lda' in step_name:
            return 'selector'
        elif 'classif' in step_name:
            return 'classifier'
        elif 'regress' in step_name:
            return 'regressor'
        return 'other'
    
    def pipeline_to_graph(self, 
                        pipeline: Pipeline, 
                        target: Optional[float] = None) -> Data:
        """
        Convert a pipeline to a graph representation.
        
        Args:
            pipeline: The pipeline to convert.
            target: Optional target value for the pipeline.
            
        Returns:
            A PyTorch Geometric Data object representing the pipeline.
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each step
        for i, step in enumerate(pipeline.steps):
            # Create node features
            step_type_embedding = self._get_step_type_embedding(step)
            
            # Add node with features
            G.add_node(i, x=step_type_embedding)
            
            # Add edges from previous nodes (fully connected to previous steps)
            for j in range(i):
                G.add_edge(j, i)
        
        # Convert to PyTorch Geometric Data object
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        x = torch.stack([data['x'] for _, data in G.nodes(data=True)])
        
        # Create target tensor if provided
        y = torch.tensor([target], dtype=torch.float) if target is not None else None
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def batch_pipelines(self, 
                      pipelines: List[Pipeline], 
                      targets: Optional[List[float]] = None) -> List[Data]:
        """
        Convert multiple pipelines to graph representations.
        
        Args:
            pipelines: List of pipelines to convert.
            targets: Optional list of target values.
            
        Returns:
            List of PyTorch Geometric Data objects.
        """
        if targets is None:
            targets = [None] * len(pipelines)
            
        return [
            self.pipeline_to_graph(pipe, target)
            for pipe, target in zip(pipelines, targets)
        ]


def create_dummy_pipeline(step_count: int = 5) -> Pipeline:
    """Create a dummy pipeline for testing."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    
    pipeline = Pipeline()
    pipeline.add_step('imputer', SimpleImputer())
    pipeline.add_step('scaler', StandardScaler())
    
    if step_count > 2:
        from sklearn.decomposition import PCA
        pipeline.add_step('pca', PCA(n_components=0.95))
    
    if step_count > 3:
        from sklearn.feature_selection import SelectKBest, f_classif
        pipeline.add_step('selector', SelectKBest(f_classif, k=10))
    
    pipeline.add_step('classifier', RandomForestClassifier())
    return pipeline
