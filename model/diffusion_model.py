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
        b = x.shape[0]  # batch size
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        # Handle case where input doesn't have batch dimension
        head_dim = q.shape[-1] // h
        
        # Ensure tensors have proper dimensions for multi-head attention
        q = q.view(b, -1, h, head_dim).permute(0, 2, 1, 3)  # [b, h, seq_len_q, head_dim]
        k = k.view(b, -1, h, head_dim).permute(0, 2, 1, 3)  # [b, h, seq_len_k, head_dim]
        v = v.view(b, -1, h, head_dim).permute(0, 2, 1, 3)  # [b, h, seq_len_v, head_dim]
        
        # Compute attention scores
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # Attention weights
        attn = F.softmax(sim, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Reshape back
        out = out.permute(0, 2, 1, 3).contiguous().view(b, -1, h * head_dim)
        
        return self.to_out(out.squeeze(1) if out.size(1) == 1 else out)


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
        
        from equivariant_gnn import EquivariantGraphConv
        
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
        try:
            # Check inputs for NaN values
            if torch.isnan(x).any():
                print("NaN detected in input x")
            if torch.isnan(edge_attr).any():
                print("NaN detected in edge_attr")
            if torch.isnan(pos).any():
                print("NaN detected in pos")
            
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
            
            # Expand condition embeddings for all nodes
            # First, get number of nodes from h
            num_nodes = h.size(0)
            nodes_per_graph = num_nodes // batch.max().item() + 1 if batch is not None else num_nodes
            
            # Create expanded condition features that match each node
            if batch is not None:
                # Map each node to its corresponding condition vector
                expanded_cond = cond_emb[batch]
            else:
                # All nodes share the same condition
                expanded_cond = cond_emb.repeat(num_nodes, 1)
            
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
                
                # Ensure time_influence has the right shape by expanding to match h_update
                if time_influence.dim() < h_update.dim():
                    # Add dimensionality to match h_update
                    time_influence = time_influence.unsqueeze(-1).expand(-1, h_update.size(-1))
                elif time_influence.size(-1) != h_update.size(-1):
                    # Reshape time_influence to match h_update's last dimension
                    time_influence = time_influence.view(time_influence.size(0), h_update.size(-1))
                
                h_update = h_update + time_influence
                
                # Residual connection
                h = h + h_update
            
            # Predict node features and positions
            node_pred = self.node_pred(h)
            pos_pred = self.pos_pred(h)
            
            # Check outputs for NaN values
            if torch.isnan(node_pred).any():
                print("NaN detected in node_pred")
                # Replace NaNs with zeros
                node_pred = torch.where(torch.isnan(node_pred), torch.zeros_like(node_pred), node_pred)
            if torch.isnan(pos_pred).any():
                print("NaN detected in pos_pred")
                # Replace NaNs with zeros
                pos_pred = torch.where(torch.isnan(pos_pred), torch.zeros_like(pos_pred), pos_pred)
            
            return node_pred, pos_pred
        except Exception as e:
            print(f"Error in CrystalDiffusionModel.forward: {e}")
            # Return zero tensors as fallback
            return torch.zeros_like(x), torch.zeros_like(pos)


class DiffusionProcess:
    """
    Diffusion process for crystal structure generation.
    
    This class implements the forward and reverse diffusion processes
    for the generation of crystal structures.
    """
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, beta_schedule='linear'):
        """
        Initialize the diffusion process.
        
        Args:
            timesteps: Number of diffusion timesteps
            beta_start: Starting noise schedule
            beta_end: Ending noise schedule
            beta_schedule: Type of noise schedule ('linear' or 'cosine')
        """
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Set up noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Pre-compute diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )
    
    def _extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps and broadcast to batch dimension."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: Initial state (x_0)
            t: Timesteps
            noise: Optional pre-generated noise
            
        Returns:
            Noised sample x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Check for NaN in input
        if torch.isnan(x_start).any():
            print("NaN detected in x_start during q_sample")
        if torch.isnan(noise).any():
            print("NaN detected in noise during q_sample")
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Check extracted values
        if torch.isnan(sqrt_alphas_cumprod_t).any():
            print("NaN detected in sqrt_alphas_cumprod_t")
        if torch.isnan(sqrt_one_minus_alphas_cumprod_t).any():
            print("NaN detected in sqrt_one_minus_alphas_cumprod_t")
            
        # Apply forward process
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Check output
        if torch.isnan(x_t).any():
            print("NaN detected in x_t output of q_sample")
            
        return x_t
    
    def loss_function(self, model, x_0, topo_cond, stab_cond, sust_cond, edge_index, edge_attr, batch_idx):
        """
        Compute diffusion loss.
        
        Args:
            model: Crystal diffusion model
            x_0: Initial state (atom features + positions)
            topo_cond: Topological conditions
            stab_cond: Stability conditions
            sust_cond: Sustainability conditions
            edge_index: Edge indices
            edge_attr: Edge attributes
            batch_idx: Batch indices
            
        Returns:
            Diffusion loss
        """
        # Check for NaNs in input
        if torch.isnan(x_0).any():
            print("NaN detected in x_0 during loss_function")

        # Get batch size and feature dimensions
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # Generate random noise
        noise = torch.randn_like(x_0)
        
        # Check for NaNs in generated noise
        if torch.isnan(noise).any():
            print("NaN detected in noise during loss_function")
        
        # Apply forward diffusion process
        x_t = self.q_sample(x_0, t, noise)
        
        # Check for NaNs after forward diffusion
        if torch.isnan(x_t).any():
            print("NaN detected in x_t after q_sample")
        
        # Model predicts noise (or x_0)
        with torch.set_grad_enabled(True):
            try:
                model_output = model(
                    x_t, t, topo_cond, stab_cond, sust_cond,
                    edge_index, edge_attr, batch_idx
                )
                
                # Check for NaNs in model output
                if torch.isnan(model_output).any():
                    print("NaN detected in model_output during loss_function")
                    return torch.tensor(float('nan'), device=device, requires_grad=True)
                
                # Compute MSE loss between predicted and actual noise
                loss = F.mse_loss(model_output, noise)
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    print("NaN detected in final loss calculation")
                    return torch.tensor(float('nan'), device=device, requires_grad=True)
                
                return loss
            except Exception as e:
                print(f"Error in diffusion loss_function: {e}")
                return torch.tensor(float('nan'), device=device, requires_grad=True)
    
    def p_mean_variance(self, model, x_t, t, topo_cond, stab_cond, sust_cond, 
                        edge_index, edge_attr, batch_idx, clip_denoised=True):
        """
        Get mean and variance for the reverse process.
        
        Args:
            model: Crystal diffusion model
            x_t: Current state at timestep t
            t: Current timestep
            topo_cond: Topological conditions
            stab_cond: Stability conditions
            sust_cond: Sustainability conditions
            edge_index: Edge indices
            edge_attr: Edge attributes
            batch_idx: Batch indices
            clip_denoised: Whether to clip the predicted denoised sample
            
        Returns:
            Tuple of (mean, variance, log_variance, predicted_x0)
        """
        # Check for NaNs in input
        if torch.isnan(x_t).any():
            print("NaN detected in x_t during p_mean_variance")
        
        # Predict noise or x_0
        try:
            model_output = model(
                x_t, t, topo_cond, stab_cond, sust_cond,
                edge_index, edge_attr, batch_idx
            )
            
            # Check for NaNs in model output
            if torch.isnan(model_output).any():
                print("NaN detected in model_output during p_mean_variance")
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            # Return dummy values if model fails
            return (
                torch.zeros_like(x_t),
                torch.zeros_like(x_t),
                torch.zeros_like(x_t),
                torch.zeros_like(x_t)
            )
        
        # The model predicts the noise added
        # Compute predicted x_0 from x_t and predicted noise
        pred_noise = model_output
        
        # Check for NaNs in predicted noise
        if torch.isnan(pred_noise).any():
            print("NaN detected in pred_noise")
        
        # Extract required diffusion parameters for timestep t
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_t = self._extract(self.alphas, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        
        # Check extracted parameters
        if torch.isnan(alpha_cumprod_t).any() or torch.isnan(alpha_t).any() or torch.isnan(beta_t).any():
            print("NaN detected in extracted diffusion parameters")
        
        # Compute predicted x_0
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Check predicted x_0
        if torch.isnan(pred_x0).any():
            print("NaN detected in pred_x0")
        
        # Clip x_0 if requested
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute posterior mean and variance
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0 +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        
        # Check posterior mean
        if torch.isnan(posterior_mean).any():
            print("NaN detected in posterior_mean")
        
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        # Check posterior variance
        if torch.isnan(posterior_variance).any() or torch.isnan(posterior_log_variance).any():
            print("NaN detected in posterior variance")
        
        return posterior_mean, posterior_variance, posterior_log_variance, pred_x0
    
    def p_sample(self, model, x_t, t, topo_cond, stab_cond, sust_cond, 
                 edge_index, edge_attr, batch_idx, clip_denoised=True):
        """
        Sample from the reverse process at timestep t.
        
        Args:
            model: Crystal diffusion model
            x_t: Current state at timestep t
            t: Current timestep
            topo_cond: Topological conditions
            stab_cond: Stability conditions
            sust_cond: Sustainability conditions
            edge_index: Edge indices
            edge_attr: Edge attributes
            batch_idx: Batch indices
            clip_denoised: Whether to clip the predicted denoised sample
            
        Returns:
            Sample at timestep t-1
        """
        # Get model mean and variance
        try:
            mean, _, log_variance, pred_x0 = self.p_mean_variance(
                model, x_t, t, topo_cond, stab_cond, sust_cond,
                edge_index, edge_attr, batch_idx, clip_denoised
            )
            
            # Check for NaNs
            if torch.isnan(mean).any() or torch.isnan(log_variance).any() or torch.isnan(pred_x0).any():
                print("NaN detected in mean, log_variance, or pred_x0 during p_sample")
                # Use zeros as fallback
                mean = torch.zeros_like(x_t)
                log_variance = torch.zeros_like(x_t)
            
            # No noise when t == 0
            noise = torch.randn_like(x_t)
            nonzero_mask = (t > 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
            
            # Sample x_{t-1} from p(x_{t-1} | x_t)
            x_t_prev = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
            
            # Check for NaNs in output
            if torch.isnan(x_t_prev).any():
                print("NaN detected in x_t_prev during p_sample")
                # Return input as fallback
                return x_t
            
            return x_t_prev
        except Exception as e:
            print(f"Error in p_sample: {e}")
            # Return input as fallback
            return x_t
    
    def p_sample_loop(self, model, shape, topo_cond, stab_cond, sust_cond, 
                      edge_index, edge_attr, batch_idx, noise=None):
        """
        Generate samples from the model by iteratively sampling.
        
        Args:
            model: Crystal diffusion model
            shape: Shape of the sample to generate
            topo_cond: Topological conditions
            stab_cond: Stability conditions
            sust_cond: Sustainability conditions
            edge_index: Edge indices
            edge_attr: Edge attributes
            batch_idx: Batch indices
            noise: Optional initial noise
            
        Returns:
            Generated sample(s)
        """
        device = edge_index.device
        
        # Start from pure noise
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise
        
        # Check for NaNs in initial noise
        if torch.isnan(x_t).any():
            print("NaN detected in initial x_t during p_sample_loop")
            # Use random uniform values as fallback
            x_t = torch.rand(shape, device=device) * 2 - 1
        
        # Iteratively denoise
        for t in tqdm(reversed(range(0, self.timesteps)), desc='Sampling'):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            try:
                with torch.no_grad():
                    x_t = self.p_sample(
                        model, x_t, t_batch, topo_cond, stab_cond, sust_cond,
                        edge_index, edge_attr, batch_idx
                    )
                
                # Check for NaNs after each step
                if torch.isnan(x_t).any():
                    print(f"NaN detected in x_t at timestep {t}")
                    # Replace NaNs with zeros
                    x_t = torch.where(torch.isnan(x_t), torch.zeros_like(x_t), x_t)
            except Exception as e:
                print(f"Error at timestep {t} in p_sample_loop: {e}")
        
        return x_t