"""Test GNN predictor with synthetic pipeline data."""
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c60.gnn import PipelinePredictor, train_gnn_predictor, evaluate_gnn_predictor
from c60.gnn.pipeline_to_graph import PipelineGraphConverter, create_dummy_pipeline

def generate_synthetic_pipelines(
    num_pipelines: int = 200,
    min_steps: int = 3,
    max_steps: int = 8
) -> tuple:
    """Generate synthetic pipelines with random performance scores."""
    pipelines = []
    targets = []
    
    for _ in range(num_pipelines):
        # Random number of steps
        num_steps = np.random.randint(min_steps, max_steps + 1)
        
        # Create pipeline with random number of steps
        pipeline = create_dummy_pipeline(step_count=num_steps)
        
        # Generate random performance score (0.0 to 1.0)
        # More complex pipelines might have better performance but higher variance
        base_score = 0.3 + 0.5 * (num_steps / max_steps)
        score = np.clip(np.random.normal(base_score, 0.1), 0, 1)
        
        pipelines.append(pipeline)
        targets.append(score)
    
    return pipelines, np.array(targets)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic pipelines...")
    pipelines, targets = generate_synthetic_pipelines(num_pipelines=500)
    
    # Initialize graph converter
    converter = PipelineGraphConverter()
    
    # Convert pipelines to graphs
    print("Converting pipelines to graphs...")
    graphs = converter.batch_pipelines(pipelines, targets)
    
    # Split into train/val/test
    train_data = graphs[:350]
    val_data = graphs[350:425]
    test_data = graphs[425:]
    
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
    print("\nTraining GNN predictor on pipeline data...")
    history = train_gnn_predictor(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=100,
        lr=1e-3,
        batch_size=16,
        weight_decay=1e-5
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
        torch.save(model.state_dict(), "models/pipeline_gnn_predictor.pt")
        print("\nModel saved successfully!")
    else:
        print("\nModel performance below threshold (R² > 0.7 required).")

if __name__ == "__main__":
    main()
