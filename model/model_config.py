"""
Model configuration and initialization utilities.

This module provides configuration options and initialization functions
for the diffusion model and related components.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """
    Configuration for the crystal diffusion model.
    """
    # Graph representation parameters
    node_dim: int = 12              # Dimension of node features
    edge_dim: int = 20              # Dimension of edge features
    max_neighbors: int = 12         # Maximum number of neighbors per atom
    cutoff_radius: float = 6.0      # Cutoff radius for neighbor search
    
    # Model architecture parameters
    hidden_dim: int = 128           # Hidden dimension
    time_emb_dim: int = 64          # Time embedding dimension
    cond_dim: int = 64              # Conditional embedding dimension
    num_gnn_layers: int = 6         # Number of GNN layers
    dropout: float = 0.1            # Dropout rate
    
    # Topological condition parameters
    topo_dim: int = 7               # Dimension of topological features
    topo_embed_dim: int = 32        # Embedding dimension for topological features
    
    # Stability condition parameters
    stab_dim: int = 2               # Dimension of stability features
    stab_embed_dim: int = 16        # Embedding dimension for stability features
    
    # Sustainability condition parameters
    sust_dim: int = 3               # Dimension of sustainability features
    sust_embed_dim: int = 16        # Embedding dimension for sustainability features
    
    # Diffusion process parameters
    timesteps: int = 1000           # Number of diffusion timesteps
    beta_start: float = 1e-4        # Starting noise level
    beta_end: float = 2e-2          # Ending noise level
    
    # Loss function parameters
    lambda_diffusion: float = 1.0   # Weight for diffusion loss
    lambda_topological: float = 1.0 # Weight for topological loss
    lambda_stability: float = 1.0   # Weight for stability loss
    lambda_sustainability: float = 1.0  # Weight for sustainability loss
    lambda_validity: float = 1.0    # Weight for validity loss
    
    # Training parameters
    learning_rate: float = 1e-4     # Learning rate
    weight_decay: float = 1e-6      # Weight decay
    batch_size: int = 32            # Batch size
    num_epochs: int = 100           # Number of epochs
    checkpoint_interval: int = 10   # Save checkpoint every N epochs
    
    # Hardware parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use
    num_workers: int = 4            # Number of data loading workers
    pin_memory: bool = True         # Pin memory for faster data transfer
    
    # Sampling parameters
    num_samples: int = 10           # Number of samples to generate
    temperature: float = 1.0        # Sampling temperature
    clip_denoised: bool = True      # Whether to clip denoised samples
    
    # Paths
    data_dir: str = "../data"       # Directory with input data
    output_dir: str = "../output"   # Directory for output files
    checkpoint_dir: str = "../checkpoints"  # Directory for model checkpoints


def init_model(config: ModelConfig):
    """
    Initialize the diffusion model and its components.
    
    Args:
        config: Model configuration
        
    Returns:
        Dictionary containing the initialized model and related components
    """
    from diffusion_model import CrystalDiffusionModel, DiffusionProcess
    from loss_functions import CombinedLoss
    from property_predictors import PropertyPredictorEnsemble
    
    # Initialize crystal diffusion model
    model = CrystalDiffusionModel(
        node_dim=config.node_dim,
        edge_dim=config.edge_dim,
        hidden_dim=config.hidden_dim,
        time_emb_dim=config.time_emb_dim,
        cond_dim=config.cond_dim,
        num_layers=config.num_gnn_layers,
        dropout=config.dropout
    )
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end
    )
    
    # Initialize property predictor ensemble
    property_predictor = PropertyPredictorEnsemble(
        node_dim=config.node_dim,
        edge_dim=config.edge_dim,
        hidden_dim=config.hidden_dim
    )
    
    # Initialize loss function
    loss_fn = CombinedLoss(
        lambda_diffusion=config.lambda_diffusion,
        lambda_topological=config.lambda_topological,
        lambda_stability=config.lambda_stability,
        lambda_sustainability=config.lambda_sustainability,
        lambda_validity=config.lambda_validity
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.learning_rate * 0.01
    )
    
    # Move to specified device
    device = torch.device(config.device)
    model.to(device)
    property_predictor.to(device)
    
    return {
        "model": model,
        "diffusion": diffusion,
        "property_predictor": property_predictor,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "config": config,
        "device": device
    }


def save_checkpoint(state: Dict[str, Any], filename: str):
    """
    Save model checkpoint.
    
    Args:
        state: State dictionary to save
        filename: Path to save the checkpoint
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename: str, map_location: Optional[str] = None):
    """
    Load model checkpoint.
    
    Args:
        filename: Path to the checkpoint
        map_location: Optional device mapping
        
    Returns:
        Loaded state dictionary
    """
    return torch.load(filename, map_location=map_location)