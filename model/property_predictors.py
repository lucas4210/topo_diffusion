"""
Property predictors for generated crystal structures.

This module implements models to predict various properties of crystal structures,
including topological invariants, stability, and sustainability metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool


class MessagePassingLayer(MessagePassing):
    """
    Base message passing layer for crystal graph property prediction.
    """
    def __init__(self, in_channels, out_channels, edge_dim=None):
        super(MessagePassingLayer, self).__init__(aggr='add')
        
        # Node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        
        # Edge feature transformation
        if edge_dim is not None:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_dim, 2 * out_channels),
                nn.ReLU(),
                nn.Linear(2 * out_channels, out_channels)
            )
        else:
            self.edge_mlp = None
            
        # Combine messages
        self.combine = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass of the message passing layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            
        Returns:
            Updated node features
        """
        # Apply message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr=None):
        """
        Message function for message passing.
        
        Args:
            x_j: Features of source nodes
            edge_attr: Edge features
            
        Returns:
            Messages to be aggregated
        """
        # Transform node features
        message = self.node_mlp(x_j)
        
        # Incorporate edge features if available
        if edge_attr is not None and self.edge_mlp is not None:
            edge_embedding = self.edge_mlp(edge_attr)
            message = message * edge_embedding
            
        return message
    
    def update(self, aggr_out, x):
        """
        Update function after message aggregation.
        
        Args:
            aggr_out: Aggregated messages
            x: Original node features
            
        Returns:
            Updated node features
        """
        # Combine original features with aggregated messages
        combined = torch.cat([x, aggr_out], dim=1)
        out = self.combine(combined)
        
        return out


class TopologicalPredictor(nn.Module):
    """
    Model to predict topological properties of crystal structures.
    """
    def __init__(self, 
                 node_dim=12,        # Node feature dimension
                 edge_dim=20,         # Edge feature dimension
                 hidden_dim=64,       # Hidden dimension
                 output_dim=7,        # Output dimension for topological features
                 num_layers=3):       # Number of message passing layers
        super().__init__()
        
        # Initial node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Initial edge embedding
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
        # Global pooling readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Output layers for topological properties
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, data):
        """
        Forward pass of the topological predictor.
        
        Args:
            data: Graph data object with node features, edge indices, and edge features
            
        Returns:
            Predicted topological features
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Embed node and edge features
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Apply message passing layers
        for mp_layer in self.mp_layers:
            x = mp_layer(x, edge_index, edge_attr)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        pooled = global_mean_pool(x, batch)
        
        # Apply readout and output layers
        readout = self.readout(pooled)
        out = self.output(readout)
        
        return out


class StabilityPredictor(nn.Module):
    """
    Model to predict stability properties (formation energy, hull distance) of crystal structures.
    """
    def __init__(self, 
                 node_dim=12,        # Node feature dimension
                 edge_dim=20,         # Edge feature dimension
                 hidden_dim=64,       # Hidden dimension
                 output_dim=2,        # Output dimension (formation energy, hull distance)
                 num_layers=3):       # Number of message passing layers
        super().__init__()
        
        # Initial node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Initial edge embedding
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
        # Global pooling readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Output layers for stability properties
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, data):
        """
        Forward pass of the stability predictor.
        
        Args:
            data: Graph data object with node features, edge indices, and edge features
            
        Returns:
            Predicted stability features [formation_energy, hull_distance]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Embed node and edge features
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Apply message passing layers
        for mp_layer in self.mp_layers:
            x = mp_layer(x, edge_index, edge_attr)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        pooled = global_mean_pool(x, batch)
        
        # Apply readout and output layers
        readout = self.readout(pooled)
        out = self.output(readout)
        
        return out


class SustainabilityPredictor(nn.Module):
    """
    Model to predict sustainability metrics of crystal structures.
    """
    def __init__(self, 
                 node_dim=12,        # Node feature dimension
                 hidden_dim=64,       # Hidden dimension
                 output_dim=3,        # Output dimension (abundance, toxicity, recyclability)
                 num_layers=2):       # Number of linear layers
        super().__init__()
        
        # For sustainability, we primarily care about the elements present,
        # so we use a simpler model that works directly with element features
        
        # Element feature extraction
        self.element_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layers for sustainability prediction
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        """
        Forward pass of the sustainability predictor.
        
        Args:
            data: Graph data object with node features
            
        Returns:
            Predicted sustainability metrics [abundance, toxicity, recyclability]
        """
        x = data.x
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Extract element features
        elem_features = self.element_embedding(x)
        
        # Global average pooling (element-wise average)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        pooled = global_mean_pool(elem_features, batch)
        
        # Apply MLP and output layer
        hidden = self.mlp(pooled)
        out = self.output(hidden)
        
        # Apply specific activation functions for each metric
        abundance_score = torch.sigmoid(out[:, 0])  # 0 to 1, higher is better
        toxicity_score = torch.sigmoid(out[:, 1])   # 0 to 1, lower is better
        recyclability_score = torch.sigmoid(out[:, 2])  # 0 to 1, higher is better
        
        return torch.stack([abundance_score, toxicity_score, recyclability_score], dim=1)


class PropertyPredictorEnsemble(nn.Module):
    """
    Ensemble of property predictors for comprehensive crystal structure evaluation.
    """
    def __init__(self, 
                 node_dim=12,        # Node feature dimension
                 edge_dim=20,         # Edge feature dimension
                 hidden_dim=64):      # Hidden dimension
        super().__init__()
        
        # Initialize individual predictors
        self.topological_predictor = TopologicalPredictor(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim
        )
        
        self.stability_predictor = StabilityPredictor(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim
        )
        
        self.sustainability_predictor = SustainabilityPredictor(
            node_dim=node_dim,
            hidden_dim=hidden_dim
        )
        
    def forward(self, data):
        """
        Forward pass of the property predictor ensemble.
        
        Args:
            data: Graph data object with node features, edge indices, and edge features
            
        Returns:
            Dictionary of predicted properties
        """
        # Get predictions from individual predictors
        topo_pred = self.topological_predictor(data)
        stab_pred = self.stability_predictor(data)
        sust_pred = self.sustainability_predictor(data)
        
        # Return all predictions as a dictionary
        return {
            'topological': topo_pred,
            'stability': stab_pred,
            'sustainability': sust_pred
        }