"""GNN-based pipeline performance predictor."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNGraphEncoder(nn.Module):
    """Graph encoder using GNN to encode pipeline DAGs."""
    
    def __init__(self, 
                 node_feature_dim: int = 32,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.pool = global_mean_pool
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for graph encoding."""
        x = F.relu(self.node_encoder(x))
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
        if batch is not None:
            x = self.pool(x, batch)
            
        return x

class PipelinePredictor(nn.Module):
    """Predicts pipeline performance using GNN."""
    
    def __init__(self, 
                 node_feature_dim: int = 32,
                 hidden_dim: int = 64,
                 num_gnn_layers: int = 3,
                 num_mlp_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        # Graph encoder
        self.gnn = GNNGraphEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # MLP head for prediction
        mlp_layers = []
        in_dim = hidden_dim
        for _ in range(num_mlp_layers - 1):
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
            
        mlp_layers.append(nn.Linear(hidden_dim, 1))  # Single output for regression
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass for pipeline performance prediction."""
        # Encode graph
        x = self.gnn(data.x, data.edge_index, data.batch)
        
        # Predict performance
        out = self.mlp(x).squeeze(-1)  # [batch_size]
        return out


def train_gnn_predictor(
    model: nn.Module,
    train_data: List[Data],
    val_data: List[Data],
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, List[float]]:
    """Train the GNN predictor."""
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training
        for i in range(0, len(train_data), batch_size):
            batch = Batch.from_data_list(
                train_data[i:i + batch_size]
            ).to(device)
            
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch) / len(train_data)
        
        # Validation
        val_metrics = evaluate_gnn_predictor(model, val_data, batch_size, device)
        
        # Logging
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_r2'].append(val_metrics['r2'])
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {epoch_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val R²: {val_metrics['r2']:.4f}")
    
    return history

def evaluate_gnn_predictor(
    model: nn.Module,
    data: List[Data],
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """Evaluate the GNN predictor."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = Batch.from_data_list(
                data[i:i + batch_size]
            ).to(device)
            
            pred = model(batch)
            loss = criterion(pred, batch.y)
            
            total_loss += loss.item() * len(batch) / len(data)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    # Calculate R² score
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'loss': total_loss,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }
