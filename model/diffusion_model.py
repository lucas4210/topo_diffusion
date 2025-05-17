"""
Diffusion model for crystal structure generation.

This module implements a diffusion-based generative model for crystal structures,
with conditional generation capabilities for controlling topological properties,
stability, and sustainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Generate position embeddings for diffusion timesteps.
        
        Args:
            time: Diffusion timestep tensor of shape [batch_size]
            
        Returns:
            Position embeddings of shape [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalEmbedding(nn.Module):
    """
    Embedding for conditional inputs to the diffusion model.
    """
    def __init__(self, 
                 topo_dim=7,        # Dimension of topological features
                 topo_embed_dim=32,  # Embedding dimension for topological features
                 stab_dim=2,         # Dimension of stability features
                 stab_embed_dim=16,  # Embedding dimension for stability features
                 sust_dim=3,         # Dimension of sustainability features
                 sust_embed_dim=16,  # Embedding dimension for sustainability features
                 combined_dim=64):   # Combined embedding dimension
        super().__init__()
        
        # Embeddings for different condition types
        self.topo_embed = nn.Sequential(
            nn.Linear(topo_dim, topo_embed_dim),
            nn.SiLU(),
            nn.Linear(topo_embed_dim, topo_embed_dim)
        )
        
        self.stab_embed = nn.Sequential(
            nn.Linear(stab_dim, stab_embed_dim),
            nn.SiLU(),
            nn.Linear(stab_embed_dim, stab_embed_dim)
        )
        
        self.sust_embed = nn.Sequential(
            nn.Linear(sust_dim, sust_embed_dim),
            nn.SiLU(),
            nn.Linear(sust_embed_dim, sust_embed_dim)
        )
        
        # Combine embeddings
        self.combined = nn.Sequential(
            nn.Linear(topo_embed_dim + stab_embed_dim + sust_embed_dim, combined_dim),
            nn.SiLU(),
            nn.Linear(combined_dim, combined_dim)
        )
        
    def forward(self, topo_cond, stab_cond, sust_cond):
        """
        Generate embeddings for conditional inputs.
        
        Args:
            topo_cond: Topological condition tensor
            stab_cond: Stability condition tensor
            sust_cond: Sustainability condition tensor
            
        Returns:
            Combined condition embedding
        """
        topo_embedding = self.topo_embed(topo_cond)
        stab_embedding = self.stab_embed(stab_cond)
        sust_embedding = self.sust_embed(sust_cond)
        
        # Concatenate embeddings
        concat_embedding = torch.cat([topo_embedding, stab_embedding, sust_embedding], dim=1)
        
        # Combine into a single embedding
        return self.combined(concat_embedding)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for integrating conditional information.
    """
    def __init__(self, query_dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(inner_dim, query_dim)
        
    def forward(self, x, context):
        """
        Apply cross-attention.
        
        Args:
            x: Query tensor
            context: Context tensor
            
        Returns:
            Output tensor after cross-attention
        """
        h = self.heads
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q, k, v = map(lambda t: t.reshape(*t.shape[:-1], h, -1).transpose(-3, -2), (q, k, v))
        
        # Compute attention scores
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # Attention weights
        attn = F.softmax(sim, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Reshape back
        out = out.transpose(-3, -2).reshape(*out.shape[:-3], -1)
        
        return self.to_out(out)


class CrystalDiffusionModel(nn.Module):
    """
    Diffusion model for generating crystal structures.
    """
    def __init__(self, 
                 node_dim=12,        # Node feature dimension
                 edge_dim=20,         # Edge feature dimension
                 hidden_dim=128,      # Hidden dimension
                 time_emb_dim=64,     # Time embedding dimension
                 cond_dim=64,         # Conditional embedding dimension
                 num_layers=6,        # Number of GNN layers
                 dropout=0.1):        # Dropout rate
        super().__init__()
        
        from models.equivariant_gnn import EquivariantGraphConv
        
        # Time step embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Node feature embeddings
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Edge feature embeddings
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # Equivariant GNN layers
        self.gnn_layers = nn.ModuleList([
            EquivariantGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Cross-attention for conditioning
        self.cond_attention = CrossAttention(hidden_dim, cond_dim)
        
        # Layer normalizations
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final MLP for node feature prediction
        self.node_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, node_dim)
        )
        
        # Position prediction network (for fractional coordinates)
        self.pos_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3D coordinates
        )
        
        # Embeddings for conditioning
        self.cond_embedding = ConditionalEmbedding(
            topo_dim=7,
            stab_dim=2,
            sust_dim=3,
            combined_dim=cond_dim
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, pos, t, 
                topo_cond, stab_cond, sust_cond, batch=None):
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            pos: Node positions (fractional coordinates)
            t: Diffusion timesteps
            topo_cond: Topological condition
            stab_cond: Stability condition
            sust_cond: Sustainability condition
            batch: Batch indices for multiple graphs
            
        Returns:
            Predicted node features and positions
        """
        # Embed node and edge features
        h = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)
        
        # Time embeddings
        time_emb = self.time_mlp(t)
        
        # Condition embeddings
        cond_emb = self.cond_embedding(topo_cond, stab_cond, sust_cond)
        
        # Expand time embeddings for each node
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        # Expand condition embeddings
        batch_size = batch.max().item() + 1
        expanded_cond = cond_emb.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, cond_emb.size(-1))
        
        # Process with GNN layers
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Apply GNN layer
            h_update = gnn(h, edge_index, edge_attr=edge_features)
            
            # Apply layer normalization
            h_update = norm(h_update)
            
            # Apply conditional cross-attention
            h_update = self.cond_attention(h_update, expanded_cond)
            
            # Apply time embedding influence
            time_influence = self.dropout(F.silu(time_emb[batch]))
            h_update = h_update + time_influence.unsqueeze(-1)
            
            # Residual connection
            h = h + h_update
        
        # Predict node features and positions
        node_pred = self.node_pred(h)
        pos_pred = self.pos_pred(h)
        
        return node_pred, pos_pred


class DiffusionProcess:
    """
    Implements the diffusion process for crystal structure generation.
    """
    def __init__(self, 
                 timesteps=1000,     # Number of diffusion timesteps
                 beta_start=1e-4,    # Starting noise level
                 beta_end=2e-2):     # Ending noise level
        """
        Initialize the diffusion process.
        
        Args:
            timesteps: Number of diffusion timesteps
            beta_start: Starting noise schedule value
            beta_end: Ending noise schedule value
        """
        self.timesteps = timesteps
        
        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x_0: Initial clean data
            t: Timesteps
            noise: Optional pre-generated noise
            
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Get alphas for timesteps
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Apply forward diffusion
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, a, t, shape):
        """Extract coefficients at specified timesteps and reshape."""
        device = t.device
        b = t.shape[0]
        out = a.to(device).gather(-1, t)
        return out.reshape(b, *((1,) * (len(shape) - 1)))
    
    def q_posterior(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0: Initial clean data
            x_t: Noisy data at timestep t
            t: Timesteps
            
        Returns:
            Posterior mean and log variance
        """
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        
        # Compute mean
        posterior_mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        
        # Extract variance
        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_log_variance_t
    
    def p_mean_variance(self, model, x_t, t, topo_cond, stab_cond, sust_cond, 
                        edge_index, edge_attr, batch=None, clip_denoised=True):
        """
        Compute the mean and variance of the diffusion posterior p(x_{t-1} | x_t).
        
        Args:
            model: Diffusion model
            x_t: Noisy data at timestep t
            t: Timesteps
            topo_cond: Topological condition
            stab_cond: Stability condition
            sust_cond: Sustainability condition
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch indices
            clip_denoised: Whether to clip the denoised signal
            
        Returns:
            Posterior mean and log variance, and predicted x_0
        """
        # Split x_t into node features and positions
        node_features, positions = torch.split(x_t, [x_t.shape[1] - 3, 3], dim=1)
        
        # Get model predictions
        pred_noise_node, pred_noise_pos = model(
            node_features, edge_index, edge_attr, positions, t,
            topo_cond, stab_cond, sust_cond, batch
        )
        
        # Combine predictions
        pred_noise = torch.cat([pred_noise_node, pred_noise_pos], dim=1)
        
        # Compute x_0 from x_t and predicted noise
        x_recon = self._predict_x0_from_noise(x_t, t, pred_noise)
        
        # Clip if requested
        if clip_denoised:
            # Clip node features and positions differently
            x_recon_node, x_recon_pos = torch.split(x_recon, [x_recon.shape[1] - 3, 3], dim=1)
            
            # Node features can be clipped to valid ranges (depends on feature encoding)
            # Positions (fractional coords) should be clipped to [0, 1]
            x_recon_pos = x_recon_pos.clamp(0., 1.)
            
            x_recon = torch.cat([x_recon_node, x_recon_pos], dim=1)
        
        # Get posterior parameters
        model_mean, model_log_variance = self.q_posterior(x_recon, x_t, t)
        
        return model_mean, model_log_variance, x_recon
    
    def _predict_x0_from_noise(self, x_t, t, noise):
        """Predict x_0 from noise."""
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, topo_cond, stab_cond, sust_cond, 
                 edge_index, edge_attr, batch=None, clip_denoised=True):
        """
        Sample from the diffusion model at timestep t.
        
        Args:
            model: Diffusion model
            x_t: Noisy data at timestep t
            t: Timesteps
            topo_cond: Topological condition
            stab_cond: Stability condition
            sust_cond: Sustainability condition
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch indices
            clip_denoised: Whether to clip the denoised signal
            
        Returns:
            Sample from p(x_{t-1} | x_t)
        """
        # Get model predictive distribution
        model_mean, model_log_variance, x_recon = self.p_mean_variance(
            model, x_t, t, topo_cond, stab_cond, sust_cond, 
            edge_index, edge_attr, batch, clip_denoised
        )
        
        # No noise when t == 0
        noise = torch.randn_like(x_t) if any(t > 0) else 0.
        
        # Compute variance
        variance = torch.exp(model_log_variance)
        
        # Sample from the posterior
        return model_mean + variance.sqrt() * noise
    
    def p_sample_loop(self, model, shape, topo_cond, stab_cond, sust_cond, 
                      edge_index, edge_attr, batch=None, noise=None, clip_denoised=True):
        """
        Generate samples from the model using the reverse diffusion process.
        
        Args:
            model: Diffusion model
            shape: Shape of the output
            topo_cond: Topological condition
            stab_cond: Stability condition
            sust_cond: Sustainability condition
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch indices
            noise: Initial noise (optional)
            clip_denoised: Whether to clip the denoised signal
            
        Returns:
            Generated samples
        """
        device = next(model.parameters()).device
        
        # Start from pure noise
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise
            
        # Iterate through all timesteps
        for t in reversed(range(0, self.timesteps)):
            # Create timestep batch
            timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Generate sample for this timestep
            x_t = self.p_sample(
                model, x_t, timesteps, topo_cond, stab_cond, sust_cond,
                edge_index, edge_attr, batch, clip_denoised
            )
            
        return x_t
    
    def sample(self, model, num_samples, topo_cond, stab_cond, sust_cond, 
               edge_index, edge_attr, batch=None, device='cuda'):
        """
        Generate multiple samples from the model.
        
        Args:
            model: Diffusion model
            num_samples: Number of samples to generate
            topo_cond: Topological condition
            stab_cond: Stability condition
            sust_cond: Sustainability condition
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch indices
            device: Device to generate samples on
            
        Returns:
            Generated samples
        """
        # Determine shape based on edge_index
        num_nodes = edge_index.max().item() + 1
        node_dim = model.node_embedding.in_features  # Get from model
        position_dim = 3  # Fractional coordinates
        
        shape = (num_nodes, node_dim + position_dim)
        
        # Move all inputs to the device
        topo_cond = topo_cond.to(device)
        stab_cond = stab_cond.to(device)
        sust_cond = sust_cond.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        if batch is not None:
            batch = batch.to(device)
        
        # Generate samples
        samples = []
        for _ in range(num_samples):
            sample = self.p_sample_loop(
                model, shape, topo_cond, stab_cond, sust_cond,
                edge_index, edge_attr, batch
            )
            samples.append(sample)
            
        return torch.stack(samples)
    
    def loss_function(self, model, x_0, topo_cond, stab_cond, sust_cond,
                      edge_index, edge_attr, batch=None):
        """
        Compute the diffusion loss for training.
        
        Args:
            model: Diffusion model
            x_0: Clean data
            topo_cond: Topological condition
            stab_cond: Stability condition
            sust_cond: Sustainability condition
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch indices
            
        Returns:
            Loss value
        """
        # Get batch size
        b = x_0.shape[0]
        device = x_0.device
        
        # Sample timesteps
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        
        # Generate noise
        noise = torch.randn_like(x_0)
        
        # Get noisy samples
        x_t = self.q_sample(x_0, t, noise)
        
        # Split x_t into node features and positions
        node_features, positions = torch.split(x_t, [x_t.shape[1] - 3, 3], dim=1)
        
        # Get model predictions
        pred_noise_node, pred_noise_pos = model(
            node_features, edge_index, edge_attr, positions, t,
            topo_cond, stab_cond, sust_cond, batch
        )
        
        # Split target noise for comparison
        target_noise_node, target_noise_pos = torch.split(noise, [noise.shape[1] - 3, 3], dim=1)
        
        # Compute losses for node features and positions
        loss_node = F.mse_loss(pred_noise_node, target_noise_node)
        loss_pos = F.mse_loss(pred_noise_pos, target_noise_pos)
        
        # Combine losses
        loss = loss_node + loss_pos
        
        return loss