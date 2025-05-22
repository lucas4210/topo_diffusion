"""
Model implementation for the Topological Materials Diffusion project.

This module contains the diffusion model architecture for generating
crystal structures with targeted properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter_mean, scatter_add
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

class GraphAttention(nn.Module):
    """
    Multi-head graph attention layer.
    
    This layer implements multi-head attention over nodes in a graph,
    with attention weights computed based on node features and edge features.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize the graph attention layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            edge_dim: Edge feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == out_dim, "out_dim must be divisible by num_heads"
        
        # Query, key, value projections
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        
        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the graph attention layer.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Get dimensions
        num_nodes = x.size(0)
        
        # Compute query, key, value projections
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        
        # Compute edge attention weights
        edge_weights = self.edge_proj(edge_attr)  # [num_edges, num_heads]
        
        # Get source and target nodes
        src, dst = edge_index
        
        # Compute attention scores
        q_dst = q[dst]  # [num_edges, num_heads, head_dim]
        k_src = k[src]  # [num_edges, num_heads, head_dim]
        
        # Compute attention scores
        scores = torch.sum(q_dst * k_src, dim=-1) * self.scale  # [num_edges, num_heads]
        
        # Add edge weights
        scores = scores + edge_weights
        
        # Compute attention weights with softmax (per destination node)
        alpha = torch.zeros_like(scores)
        for h in range(self.num_heads):
            alpha[:, h] = scatter_softmax(scores[:, h], dst)
        
        # Apply dropout to attention weights
        alpha = self.dropout(alpha)
        
        # Compute weighted sum of values
        v_src = v[src]  # [num_edges, num_heads, head_dim]
        weighted_values = v_src * alpha.unsqueeze(-1)  # [num_edges, num_heads, head_dim]
        
        # Aggregate values at destination nodes
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        for h in range(self.num_heads):
            out[:, h] = scatter_add(
                weighted_values[:, h], 
                dst, 
                dim=0, 
                dim_size=num_nodes
            )
        
        # Reshape to [num_nodes, out_dim]
        out = out.reshape(num_nodes, self.out_dim)
        
        # Apply output projection
        out = self.out_proj(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply layer normalization
        out = self.layer_norm(out)
        
        return out


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Softmax aggregation for scattered data.
    
    Args:
        src: Source tensor
        index: Index tensor
        dim: Dimension along which to perform softmax
        
    Returns:
        Softmax values
    """
    max_value_per_index = scatter_max(src, index, dim=dim)[0]
    max_per_src_element = max_value_per_index.index_select(0, index)
    
    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()
    
    sum_per_index = scatter_add(recentered_scores_exp, index, dim=dim)
    normalizing_constants = sum_per_index.index_select(0, index)
    
    return recentered_scores_exp / normalizing_constants


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Maximum aggregation for scattered data.
    
    Args:
        src: Source tensor
        index: Index tensor
        dim: Dimension along which to perform max
        
    Returns:
        Tuple of (max_values, max_indices)
    """
    max_value, _ = scatter_reduce(src, index, dim=dim, reduce="max")
    return max_value, None


def scatter_reduce(src: torch.Tensor, index: torch.Tensor, dim: int = 0, reduce: str = "sum") -> torch.Tensor:
    """
    Reduction aggregation for scattered data.
    
    Args:
        src: Source tensor
        index: Index tensor
        dim: Dimension along which to perform reduction
        reduce: Reduction operation ("sum", "mean", "max")
        
    Returns:
        Reduced tensor
    """
    if reduce == "sum":
        return scatter_add(src, index, dim=dim)
    elif reduce == "mean":
        return scatter_mean(src, index, dim=dim)
    elif reduce == "max":
        # For simplicity, we use a workaround
        dim_size = index.max().item() + 1
        result = torch.zeros(dim_size, *src.shape[1:], device=src.device)
        result.fill_(float('-inf'))
        result.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src, reduce="amax")
        result[result == float('-inf')] = 0
        return result
    else:
        raise ValueError(f"Unknown reduction: {reduce}")


class EquivariantGraphConv(MessagePassing):
    """
    Equivariant graph convolution layer.
    
    This layer implements a message passing operation that respects
    the equivariance properties of the crystal graph.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        aggr: str = "mean"
    ):
        """
        Initialize the equivariant graph convolution layer.
        
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
            aggr: Aggregation method ("mean", "sum", "max")
        """
        # Use default node_dim for torch_geometric
        super().__init__(aggr=aggr)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Node feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature projection
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(node_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the equivariant graph convolution layer.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, node_dim]
        """
        # Project node features
        h_nodes = self.node_proj(x)
        
        # Project edge features
        h_edges = self.edge_proj(edge_attr)
        
        # Debug tensor shapes
        # print(f"x shape: {x.shape}, h_nodes shape: {h_nodes.shape}")
        # print(f"edge_index shape: {edge_index.shape}, h_edges shape: {h_edges.shape}")
        
        # Propagate messages
        out = self.propagate(edge_index, x=h_nodes, edge_attr=h_edges)
        
        # Update node features
        out = self.update_net(torch.cat([x, out], dim=-1))
        
        # Residual connection
        out = x + out
        
        # Layer normalization
        out = self.layer_norm(out)
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Message function for the message passing operation.
        
        Args:
            x_i: Features of target nodes [num_edges, hidden_dim]
            x_j: Features of source nodes [num_edges, hidden_dim]
            edge_attr: Edge features [num_edges, hidden_dim]
            
        Returns:
            Messages [num_edges, hidden_dim]
        """
        # Concatenate source, target, and edge features
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Compute messages
        return self.message_net(inputs)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps.
    
    This module implements sinusoidal position embeddings for diffusion timesteps,
    similar to those used in the original DDPM paper.
    """
    
    def __init__(self, dim: int):
        """
        Initialize the sinusoidal position embeddings.
        
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sinusoidal position embeddings.
        
        Args:
            time: Diffusion timesteps [batch_size]
            
        Returns:
            Timestep embeddings [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
        
        return embeddings


class CrystalGraphDiffusionModel(nn.Module):
    """
    Diffusion model for generating crystal structures with targeted properties.
    
    This model implements a conditional diffusion process operating on crystal graphs,
    with conditioning on topological properties, stability, and sustainability metrics.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        condition_dim: int = 32
    ):
        """
        Initialize the diffusion model.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of message passing layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            condition_dim: Dimension of conditioning vectors
        """
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.condition_dim = condition_dim
        
        # Node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Time embedding for diffusion process
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.message_passing_layers = nn.ModuleList([
            EquivariantGraphConv(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            GraphAttention(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                edge_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )
        
        # Topological property prediction head
        self.topo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 5)  # 4 Z2 invariants + is_topological
        )
        
        # Stability prediction head
        self.stability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2)  # formation_energy, energy_above_hull
        )
        
        # Sustainability prediction head
        self.sustainability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)  # sustainability score
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            t: Diffusion timestep [batch_size]
            batch: Batch assignment [num_nodes]
            condition: Conditioning vectors [batch_size, condition_dim]
            cell: Unit cell parameters [batch_size, 3, 3]
            
        Returns:
            Predicted noise or velocity for the diffusion process
        """
        # Embed node features
        h = self.node_embedding(x)
        
        # Embed edge features
        edge_h = self.edge_embedding(edge_attr)
        
        # Embed timestep
        time_emb = self.time_embedding(t)
        
        # Add time embedding to node features
        h = h + time_emb[batch]
        
        # Add conditioning if provided
        if condition is not None:
            cond_emb = self.condition_embedding(condition)
            h = h + cond_emb[batch]
        
        # Apply message passing and attention layers
        for i in range(self.num_layers):
            # Message passing
            h = self.message_passing_layers[i](h, edge_index, edge_h)
            
            # Attention
            h = self.attention_layers[i](h, edge_index, edge_h, batch)
        
        # Predict noise or velocity
        pred = self.output_layers(h)
        
        return pred
    
    def predict_properties(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict material properties from the graph representation.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Dictionary of predicted properties
        """
        # Embed node features
        h = self.node_embedding(x)
        
        # Embed edge features
        edge_h = self.edge_embedding(edge_attr)
        
        # Apply message passing and attention layers
        for i in range(self.num_layers):
            # Message passing
            h = self.message_passing_layers[i](h, edge_index, edge_h)
            
            # Attention
            h = self.attention_layers[i](h, edge_index, edge_h, batch)
        
        # Pool node features to graph-level representations
        h_graph = global_mean_pool(h, batch)
        
        # Predict properties
        topo_props = self.topo_head(h_graph)
        stability_props = self.stability_head(h_graph)
        sustainability_score = self.sustainability_head(h_graph)
        
        # Extract individual properties
        z2_invariant = topo_props[:, :4]
        is_topological = torch.sigmoid(topo_props[:, 4])
        formation_energy = stability_props[:, 0]
        energy_above_hull = F.relu(stability_props[:, 1])  # Must be non-negative
        sustainability = torch.sigmoid(sustainability_score).squeeze(-1)
        
        return {
            "z2_invariant": z2_invariant,
            "is_topological": is_topological,
            "formation_energy": formation_energy,
            "energy_above_hull": energy_above_hull,
            "sustainability_score": sustainability
        }
    
    def loss_function(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        topological_props: Optional[torch.Tensor] = None,
        stability_metrics: Optional[torch.Tensor] = None,
        sustainability_scores: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss function.
        
        Args:
            pred: Model predictions
            target: Target values
            topological_props: Topological property targets
            stability_metrics: Stability metric targets
            sustainability_scores: Sustainability score targets
            loss_weights: Dictionary of loss component weights
            
        Returns:
            Dictionary of loss components and total loss
        """
        # Default loss weights
        default_weights = {
            "diffusion": 1.0,
            "topological": 0.5,
            "stability": 0.5,
            "sustainability": 0.5,
            "validity": 0.5
        }
        
        # Use provided weights or defaults
        weights = loss_weights or default_weights
        
        # Compute diffusion loss (MSE)
        diffusion_loss = F.mse_loss(pred, target)
        
        # Initialize other losses
        topological_loss = torch.tensor(0.0, device=pred.device)
        stability_loss = torch.tensor(0.0, device=pred.device)
        sustainability_loss = torch.tensor(0.0, device=pred.device)
        validity_loss = torch.tensor(0.0, device=pred.device)
        
        # Compute topological property loss if targets provided
        if topological_props is not None:
            # Extract Z2 invariants and topological classification
            z2_invariant = topological_props[:, :4]
            is_topological = topological_props[:, 4]
            
            # Predict properties from current state
            # (This is a simplified version; in practice, would use the actual predicted structure)
            pred_props = self.predict_properties(target, None, None, None)
            
            # Compute Z2 invariant loss (MSE)
            z2_loss = F.mse_loss(pred_props["z2_invariant"], z2_invariant)
            
            # Compute topological classification loss (BCE)
            topo_class_loss = F.binary_cross_entropy(
                pred_props["is_topological"],
                is_topological
            )
            
            # Combine topological losses
            topological_loss = z2_loss + topo_class_loss
        
        # Compute stability metric loss if targets provided
        if stability_metrics is not None:
            # Extract formation energy and energy above hull
            formation_energy = stability_metrics[:, 0]
            energy_above_hull = stability_metrics[:, 1]
            
            # Predict properties from current state
            pred_props = self.predict_properties(target, None, None, None)
            
            # Compute formation energy loss (MSE)
            formation_loss = F.mse_loss(
                pred_props["formation_energy"],
                formation_energy
            )
            
            # Compute energy above hull loss (MSE)
            hull_loss = F.mse_loss(
                pred_props["energy_above_hull"],
                energy_above_hull
            )
            
            # Combine stability losses
            stability_loss = formation_loss + hull_loss
        
        # Compute sustainability score loss if targets provided
        if sustainability_scores is not None:
            # Predict properties from current state
            pred_props = self.predict_properties(target, None, None, None)
            
            # Compute sustainability score loss (MSE)
            sustainability_loss = F.mse_loss(
                pred_props["sustainability_score"],
                sustainability_scores
            )
        
        # Compute validity loss (simplified)
        # In practice, would check for physical validity of the structure
        validity_loss = torch.mean(torch.abs(target)) * 0.01
        
        # Compute total loss
        total_loss = (
            weights["diffusion"] * diffusion_loss +
            weights["topological"] * topological_loss +
            weights["stability"] * stability_loss +
            weights["sustainability"] * sustainability_loss +
            weights["validity"] * validity_loss
        )
        
        # Return all loss components
        losses = {
            "diffusion_loss": diffusion_loss,
            "topological_loss": topological_loss,
            "stability_loss": stability_loss,
            "sustainability_loss": sustainability_loss,
            "validity_loss": validity_loss,
            "total_loss": total_loss
        }
        
        return losses


class DiffusionProcess:
    """
    Diffusion process for crystal graph generation.
    
    This class implements the diffusion process for generating crystal structures,
    including forward and reverse processes.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Initialize the diffusion process.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Schedule for noise level ("linear", "cosine", "quadratic")
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Set up noise schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            self._setup_cosine_schedule(num_timesteps)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Compute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Compute diffusion coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Compute posterior variance
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def _setup_cosine_schedule(self, num_timesteps: int):
        """
        Set up cosine noise schedule.
        
        Args:
            num_timesteps: Number of diffusion timesteps
        """
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from the forward diffusion process q(x_t | x_0).
        
        Args:
            x_start: Starting point (clean data) [batch_size, ...]
            t: Timesteps [batch_size]
            noise: Optional noise to add [batch_size, ...]
            
        Returns:
            Noisy samples x_t [batch_size, ...]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get diffusion coefficients for these timesteps
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        # Compute noisy samples
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start: Starting point (clean data) [batch_size, ...]
            x_t: Noisy samples at timestep t [batch_size, ...]
            t: Timesteps [batch_size]
            
        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        # Extract coefficients for these timesteps
        posterior_mean_coef1_t = extract(self.posterior_mean_coef1, t, x_start.shape)
        posterior_mean_coef2_t = extract(self.posterior_mean_coef2, t, x_start.shape)
        
        # Compute posterior mean
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        
        # Extract posterior variance and log variance
        posterior_variance_t = extract(self.posterior_variance, t, x_start.shape)
        posterior_log_variance_clipped_t = extract(
            self.posterior_log_variance_clipped, t, x_start.shape
        )
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t
    
    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        **model_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior p(x_{t-1} | x_t).
        
        Args:
            model: Noise prediction model
            x_t: Noisy samples at timestep t [batch_size, ...]
            t: Timesteps [batch_size]
            clip_denoised: Whether to clip the predicted denoised sample
            **model_kwargs: Additional arguments to the model
            
        Returns:
            Dictionary containing the posterior mean and variance
        """
        # Predict noise
        pred_noise = model(x_t, t, **model_kwargs)
        
        # Compute predicted initial sample from noise
        pred_x0 = self.predict_start_from_noise(x_t, t, pred_noise)
        
        # Clip predicted sample if requested
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute mean and variance of posterior
        model_mean, model_variance, model_log_variance = self.q_posterior_mean_variance(
            x_start=pred_x0, x_t=x_t, t=t
        )
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x0": pred_x0
        }
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        Args:
            x_t: Noisy samples at timestep t [batch_size, ...]
            t: Timesteps [batch_size]
            noise: Predicted noise [batch_size, ...]
            
        Returns:
            Predicted initial sample x_0 [batch_size, ...]
        """
        # Extract diffusion coefficients for these timesteps
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        # Compute predicted x_0
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Sample from the reverse diffusion process p(x_{t-1} | x_t).
        
        Args:
            model: Noise prediction model
            x_t: Noisy samples at timestep t [batch_size, ...]
            t: Timesteps [batch_size]
            clip_denoised: Whether to clip the predicted denoised sample
            **model_kwargs: Additional arguments to the model
            
        Returns:
            Samples x_{t-1} [batch_size, ...]
        """
        # Compute mean and variance of posterior
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, **model_kwargs
        )
        
        # Sample from posterior
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        # Compute sample
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        
        return sample
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Generate samples from the model using the reverse diffusion process.
        
        Args:
            model: Noise prediction model
            shape: Shape of the samples to generate
            device: Device to generate samples on
            noise: Optional initial noise [batch_size, ...]
            clip_denoised: Whether to clip the predicted denoised sample
            **model_kwargs: Additional arguments to the model
            
        Returns:
            Generated samples [batch_size, ...]
        """
        # Initialize with random noise
        if noise is None:
            noise = torch.randn(shape, device=device)
        
        x = noise
        
        # Iterate through all timesteps in reverse order
        for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
            # Create batch of timesteps
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Sample from reverse diffusion process
            with torch.no_grad():
                x = self.p_sample(
                    model, x, t_batch, clip_denoised=clip_denoised, **model_kwargs
                )
        
        return x
    
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        node_feature_dim: int,
        num_nodes: int,
        clip_denoised: bool = True,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            model: Noise prediction model
            batch_size: Number of samples to generate
            device: Device to generate samples on
            node_feature_dim: Dimension of node features
            num_nodes: Number of nodes in each graph
            clip_denoised: Whether to clip the predicted denoised sample
            **model_kwargs: Additional arguments to the model
            
        Returns:
            Generated samples [batch_size, num_nodes, node_feature_dim]
        """
        shape = (batch_size, num_nodes, node_feature_dim)
        return self.p_sample_loop(
            model, shape, device, clip_denoised=clip_denoised, **model_kwargs
        )


def extract(
    arr: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Extract values from a 1D tensor for a batch of indices.
    
    Args:
        arr: 1D tensor of values to extract from
        timesteps: Batch of indices into arr
        broadcast_shape: Shape to broadcast the extracted values to
        
    Returns:
        Tensor of extracted values with shape broadcast_shape
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    
    return res.expand(broadcast_shape)
