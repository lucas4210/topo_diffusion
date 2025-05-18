#MIT License

#Copyright (c) 2025 Institute of Material Science and Sustainability (IMSS)

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

"""
Equivariant Graph Neural Network for crystal graph representations.

This module implements E(3)-equivariant graph neural network layers
for processing crystal structures while preserving rotational and
translational invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class EquivariantGraphConv(MessagePassing):
    """
    Equivariant Graph Convolutional layer for crystal structures.
    
    This layer maintains E(3) equivariance by processing scalar and vector 
    features separately and ensuring that vector features transform 
    appropriately under rotations.
    """
    
    def __init__(self, scalar_in, scalar_out, vector_in=0, vector_out=0, edge_dim=None):
        """
        Initialize the equivariant graph convolutional layer.
        
        Args:
            scalar_in: Number of input scalar features
            scalar_out: Number of output scalar features
            vector_in: Number of input vector features (default: 0)
            vector_out: Number of output vector features (default: 0)
            edge_dim: Number of edge features (default: None)
        """
        super(EquivariantGraphConv, self).__init__(aggr='add')
        
        self.scalar_in = scalar_in
        self.scalar_out = scalar_out
        self.vector_in = vector_in
        self.vector_out = vector_out
        
        # Networks for scalar features
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_in + edge_dim if edge_dim else scalar_in, 2 * scalar_out),
            nn.SiLU(),
            nn.Linear(2 * scalar_out, scalar_out)
        )
        
        # Networks for vector features if needed
        if vector_in > 0 and vector_out > 0:
            self.vector_mlp = nn.Sequential(
                nn.Linear(scalar_in + edge_dim if edge_dim else scalar_in, 2 * vector_in * vector_out),
                nn.SiLU(),
                nn.Linear(2 * vector_in * vector_out, vector_in * vector_out)
            )
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset the parameters of the layer."""
        for module in self.scalar_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
        if hasattr(self, 'vector_mlp'):
            for module in self.vector_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, edge_attr=None, pos=None):
        """
        Forward pass of the equivariant graph convolutional layer.
        
        Args:
            x: Node features [scalar_features, vector_features]
            edge_index: Graph edge indices
            edge_attr: Edge features (optional)
            pos: Node positions (optional)
            
        Returns:
            Updated node features
        """
        # Split into scalar and vector features if needed
        if isinstance(x, tuple):
            x_s, x_v = x
        else:
            x_s, x_v = x, None
            
        # Process with message passing
        if x_v is not None and self.vector_in > 0:
            return self.propagate(edge_index, x=(x_s, x_v), edge_attr=edge_attr, pos=pos)
        else:
            return self.propagate(edge_index, x=x_s, edge_attr=edge_attr, pos=pos)
    
    def message(self, x_i, x_j, edge_attr, pos_i=None, pos_j=None):
        """
        Message function for message passing.
        
        Args:
            x_i: Features of target nodes
            x_j: Features of source nodes
            edge_attr: Edge features
            pos_i: Positions of target nodes (optional)
            pos_j: Positions of source nodes (optional)
            
        Returns:
            Messages to be aggregated
        """
        # Handle scalar and vector features differently
        if isinstance(x_i, tuple):
            x_s_i, x_v_i = x_i
            x_s_j, x_v_j = x_j
        else:
            x_s_i, x_v_i = x_i, None
            x_s_j, x_v_j = x_j, None
            
        # Check if position information is available
        has_positions = (pos_i is not None and pos_j is not None)
        
        if has_positions:
            # Calculate edge direction and normalize
            edge_vec = pos_j - pos_i
            edge_length = torch.norm(edge_vec, dim=1, keepdim=True) + 1e-6
            edge_direction = edge_vec / edge_length
        else:
            # Use a dummy direction if positions are not available
            if x_s_i.size(0) > 0:  # Check if there are nodes
                batch_size = x_s_i.size(0)
                edge_direction = torch.zeros(batch_size, 3, device=x_s_i.device)
        
        # Process scalar features
        if edge_attr is not None:
            message_s = self.scalar_mlp(torch.cat([x_s_j, edge_attr], dim=1))
        else:
            message_s = self.scalar_mlp(x_s_j)
            
        # Process vector features if they exist
        if x_v_j is not None and self.vector_in > 0:
            # Transform vectors based on edge direction
            if edge_attr is not None:
                trans_matrix = self.vector_mlp(torch.cat([x_s_j, edge_attr], dim=1))
            else:
                trans_matrix = self.vector_mlp(x_s_j)
                
            trans_matrix = trans_matrix.view(-1, self.vector_out, self.vector_in)
            
            # Apply transformation and ensure equivariance
            message_v = torch.matmul(trans_matrix, x_v_j.unsqueeze(-1)).squeeze(-1)
            
            # Project along edge direction for equivariance if positions are available
            if has_positions:
                edge_direction = edge_direction.unsqueeze(1)  # [N, 1, 3]
                projection = torch.sum(message_v.unsqueeze(-1) * edge_direction.unsqueeze(1), dim=-2)
                return message_s, projection
            else:
                # Skip the projection if no positions
                return message_s, message_v
        else:
            return message_s
        
    def update(self, aggr_out, x=None):
        """
        Update function after message aggregation.
        
        Args:
            aggr_out: Aggregated messages
            x: Original node features
            
        Returns:
            Updated node features
        """
        # Just return the aggregated messages
        return aggr_out