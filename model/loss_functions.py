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
        # Ensure target has right dimensions
        if target_topo_cond.dim() == 1:
            # If target is a 1D tensor, reshape to 2D
            if len(target_topo_cond) == 7:  # Single example with 7 features
                target_topo_cond = target_topo_cond.unsqueeze(0)  # Add batch dimension
            else:
                # Target is flattened: reshape based on batch size
                batch_size = predicted_topo_features.size(0)
                if len(target_topo_cond) % 7 == 0:  # Ensure it can be reshaped
                    target_topo_cond = target_topo_cond.reshape(batch_size, 7)
                else:
                    # Cannot reshape properly, use only the first 7 elements
                    print(f"Warning: Topological target shape issue, using subset of target: {target_topo_cond.shape}")
                    target_topo_cond = target_topo_cond[:7].unsqueeze(0).repeat(batch_size, 1)
        
        # Make sure both tensors have the same batch dimension
        if predicted_topo_features.size(0) != target_topo_cond.size(0):
            target_topo_cond = target_topo_cond.repeat(predicted_topo_features.size(0), 1)
        
        # Extract relevant features from predictions and targets
        # Target format: [num_topo_elements, frac_topo_elements, is_topo_spacegroup, has_inversion,
        #                avg_electronegativity, electronegativity_diff, space_group_num]
        
        # Slice indices safely to avoid dimension errors
        # Calculate Z2-related loss - focusing on features that influence Z2 invariant
        z2_idx_start = min(2, predicted_topo_features.size(1)-2)
        z2_idx_end = min(4, predicted_topo_features.size(1))
        bg_idx_start = min(4, predicted_topo_features.size(1)-2)
        bg_idx_end = min(6, predicted_topo_features.size(1))
        
        # Inversion symmetry and topo spacegroup are particularly important
        z2_loss = F.binary_cross_entropy_with_logits(
            predicted_topo_features[:, z2_idx_start:z2_idx_end], 
            target_topo_cond[:, z2_idx_start:z2_idx_end]
        )
        
        # Calculate band gap-related loss
        # Electronegativity difference strongly influences band gap
        band_gap_loss = F.mse_loss(
            predicted_topo_features[:, bg_idx_start:bg_idx_end], 
            target_topo_cond[:, bg_idx_start:bg_idx_end]
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
        # Ensure target has right dimensions
        if target_stability.dim() == 1:
            # If target is a 1D tensor, reshape to 2D
            if len(target_stability) == 2:  # Single example with 2 features
                target_stability = target_stability.unsqueeze(0)  # Add batch dimension
            else:
                # Target is flattened: reshape based on batch size
                batch_size = predicted_stability.size(0)
                if len(target_stability) % 2 == 0:  # Ensure it can be reshaped
                    target_stability = target_stability.reshape(batch_size, 2)
                else:
                    # Cannot reshape properly, use only the first 2 elements
                    print(f"Warning: Stability target shape issue, using subset of target: {target_stability.shape}")
                    target_stability = target_stability[:2].unsqueeze(0).repeat(batch_size, 1)
        
        # Make sure both tensors have the same batch dimension
        if predicted_stability.size(0) != target_stability.size(0):
            target_stability = target_stability.repeat(predicted_stability.size(0), 1)
            
        # Calculate formation energy loss
        # Lower formation energy is better
        if predicted_stability.size(1) > 0 and target_stability.size(1) > 0:
            formation_energy_loss = F.mse_loss(
                predicted_stability[:, 0], target_stability[:, 0]
            )
        else:
            formation_energy_loss = torch.tensor(0.0, device=predicted_stability.device)
        
        # Calculate hull distance loss
        # Lower hull distance is better (hull_distance > 0 is unstable)
        if predicted_stability.size(1) > 1 and target_stability.size(1) > 1:
            hull_distance_loss = F.mse_loss(
                predicted_stability[:, 1], target_stability[:, 1]
            )
            
            # Add penalty for hull_distance > 0.1 eV/atom (considered unstable)
            instability_penalty = F.relu(predicted_stability[:, 1] - 0.1).mean()
        else:
            hull_distance_loss = torch.tensor(0.0, device=predicted_stability.device)
            instability_penalty = torch.tensor(0.0, device=predicted_stability.device)
        
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
        # Ensure target has right dimensions
        if target_sustainability.dim() == 1:
            # If target is a 1D tensor, reshape to 2D
            if len(target_sustainability) == 3:  # Single example with 3 features
                target_sustainability = target_sustainability.unsqueeze(0)  # Add batch dimension
            else:
                # Target is flattened: reshape based on batch size
                batch_size = predicted_sustainability.size(0)
                if len(target_sustainability) % 3 == 0:  # Ensure it can be reshaped
                    target_sustainability = target_sustainability.reshape(batch_size, 3)
                else:
                    # Cannot reshape properly, use only the first 3 elements
                    print(f"Warning: Sustainability target shape issue, using subset of target: {target_sustainability.shape}")
                    target_sustainability = target_sustainability[:3].unsqueeze(0).repeat(batch_size, 1)
        
        # Make sure both tensors have the same batch dimension
        if predicted_sustainability.size(0) != target_sustainability.size(0):
            target_sustainability = target_sustainability.repeat(predicted_sustainability.size(0), 1)
            
        # Initialize loss components
        device = predicted_sustainability.device
        abundance_loss = torch.tensor(0.0, device=device)
        toxicity_loss = torch.tensor(0.0, device=device)
        toxicity_penalty = torch.tensor(0.0, device=device)
        recyclability_loss = torch.tensor(0.0, device=device)
        
        # Calculate abundance loss (higher abundance score is better)
        if predicted_sustainability.size(1) > 0 and target_sustainability.size(1) > 0:
            abundance_loss = F.mse_loss(
                predicted_sustainability[:, 0], target_sustainability[:, 0]
            )
        
        # Calculate toxicity loss (lower toxicity score is better)
        if predicted_sustainability.size(1) > 1 and target_sustainability.size(1) > 1:
            # Add extra penalty for exceeding target
            toxicity_loss = F.mse_loss(
                predicted_sustainability[:, 1], target_sustainability[:, 1]
            )
            toxicity_penalty = F.relu(predicted_sustainability[:, 1] - target_sustainability[:, 1]).mean()
        
        # Calculate recyclability loss (higher recyclability score is better)
        if predicted_sustainability.size(1) > 2 and target_sustainability.size(1) > 2:
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
        # Check if inputs are valid
        if positions is None or edge_index is None or edge_index.numel() == 0 or lattice is None:
            return torch.tensor(0.0, device=positions.device if positions is not None else 'cpu')
        
        # Validate lattice dimensions
        device = positions.device
        try:
            # Make sure lattice has the right shape for matrix multiplication
            if isinstance(lattice, list):
                # Convert to tensor if it's a list
                lattice = torch.tensor(lattice, device=device)
            
            # Process the lattice to ensure it's a 3x3 matrix
            lattice_matrix = None
            
            # Check lattice dimensions and handle different shapes
            if len(lattice.shape) == 2:
                if lattice.shape[0] == 3 and lattice.shape[1] == 3:
                    # Standard 3x3 lattice matrix - use as is
                    lattice_matrix = lattice
                elif lattice.shape[1] == 3:
                    # This is likely [batch_size*3, 3] shape with stacked 3x3 matrices
                    # where each 3 consecutive rows form one 3x3 matrix
                    if lattice.shape[0] % 3 == 0:
                        # Extract the first 3x3 matrix (assuming all unit cells are the same)
                        lattice_matrix = lattice[:3, :]
                    else:
                        # Handle the [N, 3] shape where N is not a multiple of 3
                        # Assume it contains atom positions - use identity matrix
                        print(f"Warning in ValidityLoss: Cannot interpret lattice shape {lattice.shape}, using identity")
                        lattice_matrix = torch.eye(3, device=device)
            elif len(lattice.shape) == 3:
                # This is a batch of lattices [batch_size, 3, 3]
                lattice_matrix = lattice[0]  # Use the first batch's lattice
            else:
                # Cannot use this lattice
                print(f"Warning in ValidityLoss: Invalid lattice shape {lattice.shape}, using identity")
                lattice_matrix = torch.eye(3, device=device)
                
            if lattice_matrix is None:
                print(f"Warning in ValidityLoss: Failed to process lattice shape {lattice.shape}, using identity")
                lattice_matrix = torch.eye(3, device=device)
                
            # Check that we have a valid 3x3 matrix
            if lattice_matrix.shape != (3, 3):
                print(f"Warning in ValidityLoss: Expected 3x3 matrix but got {lattice_matrix.shape}, using identity")
                lattice_matrix = torch.eye(3, device=device)
                
            # Calculate interatomic distances
            src, dst = edge_index
            
            # Check for index bounds issues
            max_idx = max(src.max().item(), dst.max().item())
            if max_idx >= positions.size(0):
                print(f"Warning in ValidityLoss: Index {max_idx} out of bounds for positions tensor of size {positions.size(0)}")
                return torch.tensor(0.0, device=device)
            
            # Get displacement vectors (accounting for periodic boundary conditions)
            # For simplicity, this assumes positions are already fractional coordinates
            disp = positions[dst] - positions[src]
            
            # Apply periodic boundary conditions
            disp = disp - torch.round(disp)
            
            # Convert to Cartesian coordinates - matrix multiply (n x 3) with (3 x 3)
            cart_disp = torch.matmul(disp, lattice_matrix)
            
            # Calculate distances
            distances = torch.norm(cart_disp, dim=1)
            
            # Calculate minimum distance violation loss
            distance_violation = F.relu(self.min_distance - distances).mean()
            
            # Calculate atomic overlap loss (extra penalty for severe overlap)
            overlap_violation = F.relu(0.2 - distances).mean() * 2.0
            
            # Combine losses
            total_loss = distance_violation + overlap_violation
            
            return total_loss
            
        except Exception as e:
            print(f"Warning in ValidityLoss: {e}")
            return torch.tensor(0.0, device=device)


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
        self.validity_loss = ValidityLoss()
        
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
        device = diffusion_loss.device
        
        # Initialize component losses with zero tensors
        topo_loss = torch.tensor(0.0, device=device)
        stab_loss = torch.tensor(0.0, device=device)
        sust_loss = torch.tensor(0.0, device=device)
        valid_loss = torch.tensor(0.0, device=device)
        
        # Compute component losses with error handling
        try:
            if 'topological' in predicted_properties and 'topological' in target_conditions:
                topo_loss = self.topo_loss(
                    predicted_properties['topological'], 
                    target_conditions['topological']
                )
        except Exception as e:
            print(f"Warning in topological loss: {e}")
            
        try:
            if 'stability' in predicted_properties and 'stability' in target_conditions:
                stab_loss = self.stab_loss(
                    predicted_properties['stability'], 
                    target_conditions['stability']
                )
        except Exception as e:
            print(f"Warning in stability loss: {e}")
            
        try:
            if 'sustainability' in predicted_properties and 'sustainability' in target_conditions:
                sust_loss = self.sust_loss(
                    predicted_properties['sustainability'], 
                    target_conditions['sustainability']
                )
        except Exception as e:
            print(f"Warning in sustainability loss: {e}")
            
        try:
            valid_loss = self.validity_loss(positions, edge_index, lattice)
        except Exception as e:
            print(f"Warning in validity loss: {e}")
        
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
            'topological': topo_loss.item() if isinstance(topo_loss, torch.Tensor) else float(topo_loss),
            'stability': stab_loss.item() if isinstance(stab_loss, torch.Tensor) else float(stab_loss),
            'sustainability': sust_loss.item() if isinstance(sust_loss, torch.Tensor) else float(sust_loss),
            'validity': valid_loss.item() if isinstance(valid_loss, torch.Tensor) else float(valid_loss),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components