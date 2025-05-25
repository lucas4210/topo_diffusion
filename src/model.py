"""
Crystal Graph Diffusion Model for Topological Materials Discovery.

This module defines the neural network architecture for the diffusion model.
"""

import math
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_scatter
from torch_geometric.data import Data, Batch


class CrystalGraphDiffusionModel(nn.Module):
    """
    A Graph Neural Network-based diffusion model for crystal structures.
    """
    def __init__(self, 
                 node_input_dim=64,
                 node_output_dim=64,
                 edge_feature_dim=32,
                 hidden_dim=128,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 condition_dim=0):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.node_output_dim = node_output_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.condition_dim = condition_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Graph attention layers
        self.conv_layers = nn.ModuleList([
            pyg_nn.GATv2Conv(
                hidden_dim, 
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
        # Noise prediction layers
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_output_dim)
        )
        
        # Property prediction layers
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, condition_dim)
        )
        
    def forward(self, x, edge_index, edge_attr, t, batch=None, conditions=None):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_feature_dim]
            t (torch.Tensor): Timesteps [batch_size]
            batch (torch.Tensor, optional): Batch indices for nodes. Defaults to None.
            conditions (torch.Tensor, optional): Conditioning features [batch_size, condition_dim]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Predicted noise for each node [num_nodes, node_output_dim]
                - Predicted properties for each graph [batch_size, condition_dim]
        """
        # Embed time
        t = t.view(-1, 1).float()  # [batch_size, 1]
        t_emb = self.time_mlp(t)  # [batch_size, time_dim]
        
        # Default batch indices if None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Process conditions if provided
        if conditions is not None:
            # Ensure batch indices are long tensor and get dimensions
            batch_idx = batch.long()  # [num_nodes]
            batch_size = conditions.size(0)  # [batch_size, condition_dim]
            num_nodes = x.size(0)  # [num_nodes, node_input_dim]

            # Create node-level conditions by broadcasting
            node_conditions = torch.zeros(num_nodes, self.condition_dim, device=x.device)
            for i in range(batch_size):
                mask = (batch_idx == i)
                node_conditions[mask] = conditions[i]
            
            # Concatenate conditions to node features
            x = torch.cat([x, node_conditions], dim=-1)
        else:
            # If no conditions, pad with zeros
            x = torch.cat([x, torch.zeros(x.size(0), self.condition_dim, device=x.device)], dim=-1)
            
        # Process features
        x = self.node_encoder(x)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]
        
        # Add time embeddings to node features
        x = x + t_emb[batch]
        
        # Apply graph attention layers with residual connections
        hidden = x
        for conv in self.conv_layers:
            # Graph attention
            x_out = conv(x, edge_index, edge_attr)
            # Non-linearity and dropout
            x_out = nn.functional.gelu(x_out)
            x_out = nn.functional.dropout(x_out, p=self.dropout, training=self.training)
            # Residual connection
            x = x_out + hidden
            hidden = x
            
        # Predict noise per node
        pred_noise = self.noise_predictor(x)  # [num_nodes, node_output_dim]
        
        # Pool node features for graph-level prediction using scatter mean
        graph_features = torch_scatter.scatter_mean(x, batch, dim=0)  # [batch_size, hidden_dim]
        
        # Predict properties per graph
        pred_properties = self.property_predictor(graph_features)  # [batch_size, condition_dim]
        
        return pred_noise, pred_properties

    def loss_function(self, pred_noise, target_noise, pred_properties=None, target_properties=None, property_weight=0.1):
        """
        Compute the combined loss for noise prediction and property prediction.
        
        Args:
            pred_noise (torch.Tensor): Predicted noise
            target_noise (torch.Tensor): Target noise
            pred_properties (torch.Tensor, optional): Predicted properties
            target_properties (torch.Tensor, optional): Target properties
            property_weight (float, optional): Weight for property prediction loss
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Noise prediction loss
        noise_loss = nn.functional.mse_loss(pred_noise, target_noise)
        
        # Property prediction loss if applicable
        if pred_properties is not None and target_properties is not None:
            property_loss = nn.functional.mse_loss(pred_properties, target_properties)
            return noise_loss + property_weight * property_loss
        
        return noise_loss
    
    def decode_to_structure(self, node_features):
        """
        Convert generated node features to a crystal structure.
        
        Args:
            node_features (torch.Tensor): Generated node features [num_nodes, node_output_dim]
            
        Returns:
            Structure: A pymatgen Structure object
        """
        from pymatgen.core import Structure, Lattice
        import numpy as np
        
        # Convert node features to numpy array
        node_feats = node_features.detach().cpu().numpy()
        
        # Use all node_output_dim features
        element_probs = node_feats[:, :3]
        
        # Map the probabilities to elements (simplified mapping for demonstration)
        element_mapping = {
            0: "Si",  # Silicon as default
            1: "O",   # Oxygen as common element
            2: "Al"   # Aluminum as another common element
        }
        
        # Get element indices from the highest probability
        element_indices = np.argmax(element_probs, axis=1)
        
        # Convert indices to element symbols
        species = [element_mapping[idx] for idx in element_indices]
        
        # Extract position information from the remaining features
        # Assuming last 3 dimensions encode fractional coordinates
        frac_coords = node_feats[:, 3:]
        
        # Normalize fractional coordinates to [0, 1]
        frac_coords = np.mod(frac_coords, 1.0)
        
        # Create a reasonable lattice (can be optimized later)
        lattice = Lattice.cubic(10.0)  # Default 10 Å cubic cell
        
        # Create the structure
        structure = Structure(
            lattice=lattice,
            species=species,
            coords=frac_coords,
            coords_are_cartesian=False
        )
        
        return structure


class DiffusionProcess:
    """
    Handles the diffusion process for generating crystal structures.
    Implements forward diffusion (adding noise) and reverse diffusion (denoising).
    """
    def __init__(self, 
                num_timesteps=1000,
                beta_schedule='linear',
                beta_start=1e-4,
                beta_end=2e-2):
        """
        Initialize the diffusion process.
        
        Args:
            num_timesteps (int): Number of diffusion steps
            beta_schedule (str): Schedule for noise variance ('linear' or 'cosine')
            beta_start (float): Starting value for noise schedule
            beta_end (float): Ending value for noise schedule
        """
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Set up noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = torch.clamp(self.betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]), self.alphas_cumprod[:-1]])
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([torch.tensor([self.posterior_variance[1]]), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, edge_index=None, edge_attr=None, batch=None, clip_denoised=True):
        """Sample from p(x_{t-1} | x_t)"""
        with torch.no_grad():
            # Flatten x_t if it's 3D (batch, nodes, features)
            orig_shape = x_t.shape
            if x_t.dim() == 3:
                x_t_flat = x_t.reshape(-1, x_t.shape[-1])
                # Adjust edge_index, edge_attr, batch for simple chain graph
                if edge_index is not None:
                    # Repeat edge_index for each graph in batch
                    edge_index = edge_index.repeat(1, orig_shape[0])
                    offset = torch.arange(orig_shape[0], device=x_t.device) * orig_shape[1]
                    edge_index = edge_index + offset.repeat_interleave(edge_index.shape[1]//orig_shape[0]).unsqueeze(0)
                    edge_attr = edge_attr.repeat(orig_shape[0], 1)
                    batch = torch.arange(orig_shape[0], device=x_t.device).repeat_interleave(orig_shape[1])
                else:
                    batch = None
            else:
                x_t_flat = x_t
            # Predict noise
            pred_noise, _ = model(x_t_flat, edge_index, edge_attr, t, batch=batch)
            # Reshape pred_noise back to original shape
            if x_t.dim() == 3:
                pred_noise = pred_noise.reshape(orig_shape)
            # Compute mean for posterior
            alpha = self._extract(self.alphas, t, x_t.shape)
            alpha_bar = self._extract(self.alphas_cumprod, t, x_t.shape)
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            mean = pred_x0 * torch.sqrt(alpha) + pred_noise * torch.sqrt(1 - alpha)
            # No noise when t == 0
            if t[0] == 0:
                return mean
            noise = torch.randn_like(x_t)
            variance = self._extract(self.posterior_variance, t, x_t.shape)
            log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
            return mean + torch.exp(0.5 * log_variance) * noise
    
    def p_sample_loop(self, model, shape, noise=None, condition=None, edge_index=None, edge_attr=None, batch=None, progress=True):
        """Sample from the reverse process"""
        device = next(model.parameters()).device
        
        # Initialize with noise and graph structure
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise
            
        # Create random graph structure if not provided
        if edge_index is None or edge_attr is None:
            num_nodes = shape[1]
            # Create a simple chain graph as default
            edge_index = torch.stack([
                torch.arange(num_nodes-1),
                torch.arange(1, num_nodes)
            ], dim=0).to(device)
            edge_attr = torch.ones(edge_index.size(1), model.edge_feature_dim, device=device)
        
        # Create progress bar if requested
        timesteps = list(reversed(range(self.num_timesteps)))
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc='Sampling')
        
        # Iteratively denoise
        for t in timesteps:
            # Create batch of same timestep
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Get prediction from model
            with torch.no_grad():
                x_t = self.p_sample(
                    model, 
                    x_t, 
                    t_batch, 
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    clip_denoised=True
                )
        
        return x_t
    
    def _extract(self, arr, t, broadcast_shape):
        """Extract and broadcast values from a 1-D tensor"""
        arr = arr.to(t.device)
        out = arr.gather(-1, t).reshape(t.shape[0], *((1,) * (len(broadcast_shape) - 1)))
        return out.expand(broadcast_shape[0], *broadcast_shape[1:])
    
    def to(self, device):
        """Move parameters to specified device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
    
    def cpu(self):
        """Move parameters to CPU"""
        return self.to('cpu')
    
    def cuda(self):
        """Move parameters to CUDA"""
        return self.to('cuda')


class GraphAttention(nn.Module):
    """
    Multi-head graph attention layer with edge features.
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 heads=1,
                 edge_dim=None,
                 dropout=0.0,
                 use_bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        # Linear transformations
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        if hasattr(self, 'lin_edge'):
            nn.init.xavier_uniform_(self.lin_edge.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        H, C = self.heads, self.out_channels
        
        # Linear transformations
        query = self.lin_query(x).view(-1, H, C)  # [N, H, C]
        key = self.lin_key(x).view(-1, H, C)      # [N, H, C]
        value = self.lin_value(x).view(-1, H, C)  # [N, H, C]
        
        # Add edge features if available
        if edge_attr is not None and self.edge_dim is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, H, C)  # [E, H, C]
            key_j = key[edge_index[1]]  # [E, H, C]
            key_j = key_j + edge_attr   # Add edge features to keys
        else:
            key_j = key[edge_index[1]]  # [E, H, C]
        
        # Compute attention scores
        query_i = query[edge_index[0]]  # [E, H, C]
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(C)  # [E, H]
        alpha = torch.softmax(alpha, dim=0)  # Normalize attention weights
        
        # Apply dropout to attention weights
        alpha = nn.functional.dropout(alpha, p=self.dropout, training=self.training)
        
        # Compute weighted sum
        out = value[edge_index[1]] * alpha.unsqueeze(-1)  # [E, H, C]
        out = torch_scatter.scatter_add(out, edge_index[0], dim=0)  # [N, H, C]
        out = out.view(-1, H * C)  # [N, H*C]
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class EquivariantGraphConv(nn.Module):
    """
    Equivariant graph convolution layer that preserves crystal symmetries.
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 edge_dim=None,
                 aggr='mean'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.aggr = aggr
        
        # Node feature transformation
        self.lin_node = nn.Linear(in_channels, out_channels)
        
        # Edge feature transformation
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        if hasattr(self, 'lin_edge'):
            nn.init.xavier_uniform_(self.lin_edge.weight)
    
    def forward(self, x, edge_index, edge_attr=None):
        # Transform node features
        x = self.lin_node(x)  # [N, out_channels]
        
        # Transform edge features if available
        if edge_attr is not None and self.edge_dim is not None:
            edge_attr = self.lin_edge(edge_attr)  # [E, out_channels]
            
            # Add transformed edge features to source nodes
            row, col = edge_index
            x_j = x[col]  # [E, out_channels]
            x_j = x_j + edge_attr  # Add edge features
            
            # Aggregate messages
            out = torch_scatter.scatter(x_j, row, dim=0, dim_size=x.size(0), reduce=self.aggr)
        else:
            # Simple message passing without edge features
            row, col = edge_index
            x_j = x[col]  # [E, out_channels]
            out = torch_scatter.scatter(x_j, row, dim=0, dim_size=x.size(0), reduce=self.aggr)
        
        return out


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timesteps in the diffusion process.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        """
        Args:
            time: A 1-D tensor of timesteps
            
        Returns:
            Tensor: Position embeddings of shape [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1, 0, 0))
        
        return embeddings
