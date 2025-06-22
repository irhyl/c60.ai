"""Test script for GNN-based pipeline performance predictor."""
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c60.gnn import PipelinePredictor, train_gnn_predictor, evaluate_gnn_predictor

def generate_synthetic_data(num_graphs: int = 100, 
                          num_nodes: int = 10,
                          node_feature_dim: int = 32) -> list:
    """Generate synthetic graph data for testing."""
    data_list = []
    
    for _ in range(num_graphs):
        # Random graph structure (Erdős–Rényi model)
        edge_index = torch.tensor([
            [i, j] for i in range(num_nodes) for j in range(num_nodes) 
            if np.random.rand() < 0.3 and i != j
        ], dtype=torch.long).t().contiguous()
        
        # Random node features
        x = torch.randn(num_nodes, node_feature_dim)
        
        # Random target (performance metric between 0 and 1)
        y = torch.rand(1).clamp(0, 1)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(num_graphs=200, num_nodes=20)
    
    # Split into train/val/test
    train_data = data[:140]
    val_data = data[140:170]
    test_data = data[170:]
    
    print(f"Train: {len(train_data)} graphs")
    print(f"Val: {len(val_data)} graphs")
    print(f"Test: {len(test_data)} graphs")
    
    # Initialize model
    model = PipelinePredictor(
        node_feature_dim=32,
        hidden_dim=64,
        num_gnn_layers=3,
        num_mlp_layers=2,
        dropout=0.2
    )
    
    print("\nModel architecture:")
    print(model)
    
    # Train model
    print("\nTraining GNN predictor...")
    history = train_gnn_predictor(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=50,
        lr=1e-3,
        batch_size=16
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_gnn_predictor(
        model=model,
        data=test_data,
        batch_size=16
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # Save model if R² > 0.7
    if test_metrics['r2'] > 0.7:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/gnn_predictor.pt")
        print("\nModel saved successfully!")
    else:
        print("\nModel performance below threshold (R² > 0.7 required).")

if __name__ == "__main__":
    main()
