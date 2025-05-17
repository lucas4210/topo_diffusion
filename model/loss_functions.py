"""
Multi-objective loss function for sustainable topological materials.

This module implements the combined loss function that balances topological properties,
stability, and sustainability objectives for the crystal diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalLoss(nn.Module):
    """
    Loss component that encourages the generation of materials with desired topological properties.
    """
    def __init__(self, z2_weight=1.0, band_gap_weight=0.5):
        """
        Initialize the topological loss.
        
        Args:
            z2_weight: Weight for Z2 invariant matching
            band_gap_weight: Weight for band gap matching
        """
        super().__init__()
        self.z2_weight = z2_weight
        self.band_gap_weight = band_gap_weight
        
    def forward(self, predicted_topo_features, target_topo_cond):
        """
        Compute topological loss.
        
        Args:
            predicted_topo_features: Predicted topological features
            target_topo_cond: Target topological conditions
            
        Returns:
            Topological loss value
        """
        # Extract relevant features from predictions and targets
        # Target format: [num_topo_elements, frac_topo_elements, is_topo_spacegroup, has_inversion,
        #                avg_electronegativity, electronegativity_diff, space_group_num]
        
        # Calculate Z2-related loss - focusing on features that influence Z2 invariant
        # Inversion symmetry and topo spacegroup are particularly important
        z2_loss = F.binary_cross_entropy_with_logits(
            predicted_topo_features[:, 2:4], target_topo_cond[:, 2:4]
        )
        
        # Calculate band gap-related loss
        # Electronegativity difference strongly influences band gap
        band_gap_loss = F.mse_loss(
            predicted_topo_features[:, 4:6], target_topo_cond[:, 4:6]
        )
        
        # Combine losses
        total_loss = self.z2_weight * z2_loss + self.band_gap_weight * band_gap_loss
        
        return total_loss


class StabilityLoss(nn.Module):
    """
    Loss component that encourages the generation of thermodynamically stable materials.
    """
    def __init__(self, formation_energy_weight=1.0, hull_distance_weight=1.0):
        """
        Initialize the stability loss.
        
        Args:
            formation_energy_weight: Weight for formation energy matching
            hull_distance_weight: Weight for hull distance matching
        """
        super().__init__()
        self.formation_energy_weight = formation_energy_weight
        self.hull_distance_weight = hull_distance_weight
        
    def forward(self, predicted_stability, target_stability):
        """
        Compute stability loss.
        
        Args:
            predicted_stability: Predicted stability features [formation_energy, hull_distance]
            target_stability: Target stability conditions
            
        Returns:
            Stability loss value
        """
        # Calculate formation energy loss
        # Lower formation energy is better
        formation_energy_loss = F.mse_loss(
            predicted_stability[:, 0], target_stability[:, 0]
        )
        
        # Calculate hull distance loss
        # Lower hull distance is better (hull_distance > 0 is unstable)
        hull_distance_loss = F.mse_loss(
            predicted_stability[:, 1], target_stability[:, 1]
        )
        
        # Add penalty for hull_distance > 0.1 eV/atom (considered unstable)
        instability_penalty = F.relu(predicted_stability[:, 1] - 0.1).mean()
        
        # Combine losses
        total_loss = (
            self.formation_energy_weight * formation_energy_loss + 
            self.hull_distance_weight * (hull_distance_loss + instability_penalty)
        )
        
        return total_loss


class SustainabilityLoss(nn.Module):
    """
    Loss component that encourages the generation of sustainable materials.
    """
    def __init__(self, abundance_weight=1.0, toxicity_weight=1.0, recyclability_weight=0.5):
        """
        Initialize the sustainability loss.
        
        Args:
            abundance_weight: Weight for element abundance matching
            toxicity_weight: Weight for toxicity minimization
            recyclability_weight: Weight for recyclability maximization
        """
        super().__init__()
        self.abundance_weight = abundance_weight
        self.toxicity_weight = toxicity_weight
        self.recyclability_weight = recyclability_weight
        
    def forward(self, predicted_sustainability, target_sustainability):
        """
        Compute sustainability loss.
        
        Args:
            predicted_sustainability: Predicted sustainability features [abundance, toxicity, recyclability]
            target_sustainability: Target sustainability conditions
            
        Returns:
            Sustainability loss value
        """
        # Calculate abundance loss (higher abundance score is better)
        abundance_loss = F.mse_loss(
            predicted_sustainability[:, 0], target_sustainability[:, 0]
        )
        
        # Calculate toxicity loss (lower toxicity score is better)
        # Add extra penalty for exceeding target
        toxicity_loss = F.mse_loss(
            predicted_sustainability[:, 1], target_sustainability[:, 1]
        )
        toxicity_penalty = F.relu(predicted_sustainability[:, 1] - target_sustainability[:, 1]).mean()
        
        # Calculate recyclability loss (higher recyclability score is better)
        recyclability_loss = F.mse_loss(
            predicted_sustainability[:, 2], target_sustainability[:, 2]
        )
        
        # Combine losses
        total_loss = (
            self.abundance_weight * abundance_loss + 
            self.toxicity_weight * (toxicity_loss + toxicity_penalty) + 
            self.recyclability_weight * recyclability_loss
        )
        
        return total_loss


class ValidityLoss(nn.Module):
    """
    Loss component that encourages physically valid crystal structures.
    """
    def __init__(self, min_distance=0.5):
        """
        Initialize the validity loss.
        
        Args:
            min_distance: Minimum allowed interatomic distance in Angstroms
        """
        super().__init__()
        self.min_distance = min_distance
        
    def forward(self, positions, edge_index, lattice):
        """
        Compute validity loss.
        
        Args:
            positions: Atom positions (fractional coordinates)
            edge_index: Edge indices
            lattice: Lattice parameters
            
        Returns:
            Validity loss value
        """
        # Calculate interatomic distances
        src, dst = edge_index
        
        # Get displacement vectors (accounting for periodic boundary conditions)
        # For simplicity, this assumes positions are already fractional coordinates
        disp = positions[dst] - positions[src]
        disp = disp - torch.round(disp)  # Apply periodic boundary conditions
        
        # Convert to Cartesian coordinates
        cart_disp = torch.matmul(disp, lattice)
        
        # Calculate distances
        distances = torch.norm(cart_disp, dim=1)
        
        # Calculate minimum distance violation loss
        distance_violation = F.relu(self.min_distance - distances).mean()
        
        # Calculate atomic overlap loss (extra penalty for severe overlap)
        overlap_violation = F.relu(0.2 - distances).mean() * 2.0
        
        # Combine losses
        total_loss = distance_violation + overlap_violation
        
        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for sustainable topological materials generation.
    """
    def __init__(self, 
                 lambda_diffusion=1.0,
                 lambda_topological=1.0,
                 lambda_stability=1.0,
                 lambda_sustainability=1.0,
                 lambda_validity=1.0):
        """
        Initialize the combined loss function.
        
        Args:
            lambda_diffusion: Weight for diffusion loss
            lambda_topological: Weight for topological loss
            lambda_stability: Weight for stability loss
            lambda_sustainability: Weight for sustainability loss
            lambda_validity: Weight for validity loss
        """
        super().__init__()
        
        self.lambda_diffusion = lambda_diffusion
        self.lambda_topological = lambda_topological
        self.lambda_stability = lambda_stability
        self.lambda_sustainability = lambda_sustainability
        self.lambda_validity = lambda_validity
        
        # Initialize component losses
        self.topo_loss = TopologicalLoss()
        self.stab_loss = StabilityLoss()
        self.sust_loss = SustainabilityLoss()
        self.valid_loss = ValidityLoss()
        
    def forward(self, diffusion_loss, predicted_properties, target_conditions, 
                positions, edge_index, lattice):
        """
        Compute combined loss.
        
        Args:
            diffusion_loss: Base diffusion model loss
            predicted_properties: Dictionary of predicted properties
            target_conditions: Dictionary of target conditions
            positions: Atom positions
            edge_index: Edge indices
            lattice: Lattice parameters
            
        Returns:
            Combined loss value and individual loss components
        """
        # Compute component losses
        topo_loss = self.topo_loss(
            predicted_properties['topological'], 
            target_conditions['topological']
        )
        
        stab_loss = self.stab_loss(
            predicted_properties['stability'], 
            target_conditions['stability']
        )
        
        sust_loss = self.sust_loss(
            predicted_properties['sustainability'], 
            target_conditions['sustainability']
        )
        
        valid_loss = self.valid_loss(positions, edge_index, lattice)
        
        # Combine losses
        total_loss = (
            self.lambda_diffusion * diffusion_loss +
            self.lambda_topological * topo_loss +
            self.lambda_stability * stab_loss +
            self.lambda_sustainability * sust_loss +
            self.lambda_validity * valid_loss
        )
        
        # Return total loss and individual components for logging
        loss_components = {
            'diffusion': diffusion_loss.item(),
            'topological': topo_loss.item(),
            'stability': stab_loss.item(),
            'sustainability': sust_loss.item(),
            'validity': valid_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components