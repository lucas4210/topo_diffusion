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
Crystal Graph representation module for topological materials.

This module provides utilities for converting crystal structures into graph representations
that can be used with graph neural networks, specifically designed to capture features
relevant for topological material properties.
"""

import torch
import numpy as np
from pymatgen.core import Structure
from torch_geometric.data import Data


class CrystalGraphConverter:
    """
    Converts crystal structures to graph representations.
    
    This class handles the conversion of pymatgen Structure objects to
    graph representations suitable for machine learning, with special
    attention to features that influence topological properties.
    """
    
    def __init__(self, radius=6.0, max_neighbors=12, periodic=True,
                 atom_features=None, edge_features=None):
        """
        Initialize the crystal graph converter.
        
        Args:
            radius: Cutoff radius for neighbor search
            max_neighbors: Maximum number of neighbors per atom
            periodic: Whether to use periodic boundary conditions
            atom_features: Custom atom feature function (optional)
            edge_features: Custom edge feature function (optional)
        """
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.periodic = periodic
        self.atom_features = atom_features or self._default_atom_features
        self.edge_features = edge_features or self._default_edge_features
        
        # Precompute element properties for faster access
        self._setup_element_properties()
        
    def _setup_element_properties(self):
        """Precompute element properties for faster access."""
        # Atomic number -> electronegativity, atomic radius, etc.
        self.element_props = {
            # Format: Z: [electronegativity, atomic_radius, s_valence, p_valence, d_valence, f_valence]
            1: [2.20, 0.38, 1, 0, 0, 0],  # H
            2: [0.00, 0.32, 2, 0, 0, 0],  # He
            3: [0.98, 1.52, 1, 0, 0, 0],  # Li
            4: [1.57, 1.12, 2, 0, 0, 0],  # Be
            5: [2.04, 0.85, 2, 1, 0, 0],  # B
            6: [2.55, 0.75, 2, 2, 0, 0],  # C
            7: [3.04, 0.71, 2, 3, 0, 0],  # N
            8: [3.44, 0.63, 2, 4, 0, 0],  # O
            # ... Add more elements as needed
            # For the complete dataset you would include all elements
        }
        
        # Default values for missing elements
        self.default_props = [2.0, 1.0, 0, 0, 0, 0]
        
    def _default_atom_features(self, element):
        """
        Generate default atom features.
        
        Args:
            element: Element object from pymatgen
            
        Returns:
            Tensor of atom features
        """
        atomic_num = element.Z
        # Get element properties or use defaults
        props = self.element_props.get(atomic_num, self.default_props)
        
        features = [
            atomic_num,
            props[0],  # Electronegativity
            props[1],  # Atomic radius
            props[2],  # s valence
            props[3],  # p valence
            props[4],  # d valence
            props[5],  # f valence
            element.X,  # Electronegativity (Pauling)
            element.row,  # Period
            element.group,  # Group
            element.is_metal,  # Is metal
            element.is_transition_metal,  # Is transition metal
        ]
        
        return torch.tensor(features, dtype=torch.float)
    
    def _default_edge_features(self, dist, vector):
        """
        Generate default edge features.
        
        Args:
            dist: Distance between atoms
            vector: Vector between atoms
            
        Returns:
            Tensor of edge features
        """
        # Compute Gaussian basis functions
        gaussian_means = torch.linspace(0, self.radius, 20)
        gaussian_width = gaussian_means[1] - gaussian_means[0]
        
        # Distance features using Gaussian basis
        dist_tensor = torch.tensor([dist], dtype=torch.float)
        gaussian_features = torch.exp(-(dist_tensor - gaussian_means)**2 / (2 * gaussian_width**2))
        
        # Spherical harmonics features could be added here for better angular resolution
        
        return gaussian_features
    
    def convert(self, structure):
        """
        Convert a pymatgen Structure to a graph representation.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            torch_geometric.data.Data object
        """
        if not isinstance(structure, Structure):
            raise TypeError("Expected a pymatgen Structure object")
        
        # Get neighbors within cutoff radius
        num_atoms = len(structure)
        all_neighbors = structure.get_all_neighbors(self.radius, include_index=True)
        
        # Initialize node features
        node_features = []
        for site in structure.sites:
            node_features.append(self.atom_features(site.specie))
        
        # Stack node features
        if node_features:
            node_features = torch.stack(node_features)
        else:
            # Fallback if no atoms (should never happen)
            node_features = torch.zeros((0, 12), dtype=torch.float)
            
        # Get node positions (fractional coordinates)
        pos_frac = torch.tensor(structure.frac_coords, dtype=torch.float)
        # Get node positions (cartesian coordinates)
        pos_cart = torch.tensor(structure.cart_coords, dtype=torch.float)
        
        # Extract lattice information
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float)
        
        # Process edges (limit to max_neighbors per atom if needed)
        edge_indices = []
        edge_features = []
        edge_vectors = []
        
        for i, neighbors in enumerate(all_neighbors):
            # Sort neighbors by distance
            neighbors = sorted(neighbors, key=lambda x: x[1])
            
            # Limit number of neighbors if needed
            if self.max_neighbors > 0:
                neighbors = neighbors[:self.max_neighbors]
                
            for neighbor, dist, j in neighbors:
                edge_indices.append((i, j))
                
                # Get edge vector (respecting periodic boundary conditions)
                vector = structure.sites[j].coords - structure.sites[i].coords
                if self.periodic:
                    vector = structure.lattice.get_distance_and_image(
                        structure.sites[i].frac_coords,
                        structure.sites[j].frac_coords)[1]
                    vector = np.dot(vector, structure.lattice.matrix)
                
                edge_vectors.append(vector)
                edge_features.append(self.edge_features(dist, vector))
        
        # Convert to tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_features) if edge_features else None
            edge_vec = torch.tensor(edge_vectors, dtype=torch.float) if edge_vectors else None
        else:
            # Fallback for isolated atoms
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = None
            edge_vec = None
        
        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos_cart,
            frac_coords=pos_frac,
            lattice=lattice,
            edge_vec=edge_vec,
            num_nodes=num_atoms
        )
        
        return data
    
    def convert_batch(self, structures):
        """
        Convert a batch of structures to graph representations.
        
        Args:
            structures: List of pymatgen Structure objects
            
        Returns:
            List of torch_geometric.data.Data objects
        """
        return [self.convert(structure) for structure in structures]


class TopologicalFeatures:
    """
    Extract topological feature descriptors from crystal structures.
    
    This class adds additional features specifically relevant for
    topological materials prediction.
    """
    
    def __init__(self):
        """Initialize the topological features extractor."""
        # Elements often found in topological materials
        self.topo_elements = ['Bi', 'Sb', 'Te', 'Se', 'Sn', 'Pb', 'Hg', 'Tl']
        
        # Space groups that often host topological phases
        self.topo_spacegroups = [2, 164, 166, 176, 187, 189, 191, 194, 221, 224, 229]
    
    def get_features(self, structure):
        """
        Extract topological features from a structure.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Dictionary of topological features
        """
        # Calculate composition-based features
        elements = [str(el) for el in structure.composition.elements]
        num_topo_elements = sum(1 for el in elements if el in self.topo_elements)
        frac_topo_elements = num_topo_elements / len(elements) if elements else 0
        
        # Space group information
        sg_num = structure.get_space_group_info()[1]
        is_topo_spacegroup = sg_num in self.topo_spacegroups
        
        # Inversion symmetry
        has_inversion = structure.has_inversion_symmetry()
        
        # Band theory relevant descriptors
        avg_electronegativity = np.mean([el.X for el in structure.composition.elements])
        electronegativity_diff = np.max([el.X for el in structure.composition.elements]) - \
                                  np.min([el.X for el in structure.composition.elements])
        
        # Return features as a dictionary
        return {
            'num_topo_elements': num_topo_elements,
            'frac_topo_elements': frac_topo_elements,
            'is_topo_spacegroup': is_topo_spacegroup,
            'has_inversion': has_inversion,
            'avg_electronegativity': avg_electronegativity,
            'electronegativity_diff': electronegativity_diff,
            'space_group_num': sg_num
        }
    
    def enrich_graph(self, graph, structure):
        """
        Add topological features to a graph representation.
        
        Args:
            graph: torch_geometric.data.Data object
            structure: pymatgen Structure object
            
        Returns:
            Enriched graph with topological features
        """
        # Get topological features
        topo_features = self.get_features(structure)
        
        # Convert to tensor and add to graph
        topo_tensor = torch.tensor([
            topo_features['num_topo_elements'],
            topo_features['frac_topo_elements'],
            float(topo_features['is_topo_spacegroup']),
            float(topo_features['has_inversion']),
            topo_features['avg_electronegativity'],
            topo_features['electronegativity_diff'],
            topo_features['space_group_num'] / 230.0  # Normalize
        ], dtype=torch.float)
        
        # Add to graph as a global feature
        graph.topo_features = topo_tensor
        
        return graph


class SustainabilityFeatures:
    """
    Extract sustainability feature descriptors from crystal structures.
    
    This class adds additional features specifically relevant for
    sustainability assessment of materials.
    """
    
    def __init__(self, element_data=None):
        """
        Initialize the sustainability features extractor.
        
        Args:
            element_data: DataFrame with element sustainability data (optional)
        """
        # Default abundance and toxicity ratings for common elements
        # These would be replaced with actual data from the sustainability dataset
        self.default_sustainability = {
            # Element: [abundance_score, toxicity_score, recyclability_score]
            'H':  [0.95, 0.0, 0.9],
            'He': [0.5, 0.0, 0.9],
            'Li': [0.4, 0.3, 0.7],
            'Be': [0.1, 0.9, 0.3],
            'B':  [0.5, 0.3, 0.6],
            'C':  [0.9, 0.0, 0.8],
            'N':  [0.9, 0.0, 0.9],
            'O':  [0.99, 0.0, 0.9],
            # ... Add more elements as needed
        }
        
        # Default values for missing elements
        self.default_values = [0.1, 0.5, 0.3]  # Low abundance, medium toxicity, low recyclability
        
        # Load custom element data if provided
        self.element_data = element_data
    
    def get_features(self, structure):
        """
        Extract sustainability features from a structure.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Dictionary of sustainability features
        """
        elements = structure.composition.elements
        formulas = structure.composition.get_el_amt_dict()
        total_atoms = sum(formulas.values())
        
        # Initialize sustainability metrics
        abundance_scores = []
        toxicity_scores = []
        recyclability_scores = []
        
        # Calculate for each element
        for element in elements:
            symbol = str(element)
            fraction = formulas[element.symbol] / total_atoms
            
            # Get sustainability data for this element
            if self.element_data is not None and symbol in self.element_data:
                # Use provided dataset
                el_data = self.element_data[symbol]
                abundance = el_data['abundance_score']
                toxicity = el_data['toxicity_rating'] / 3.0  # Normalize to 0-1
                recyclability = el_data.get('recyclability_score', 0.5)  # Default if not available
            else:
                # Use default values
                sustainability = self.default_sustainability.get(symbol, self.default_values)
                abundance = sustainability[0]
                toxicity = sustainability[1]
                recyclability = sustainability[2]
            
            # Weight by atom fraction
            abundance_scores.append(abundance * fraction)
            toxicity_scores.append(toxicity * fraction)
            recyclability_scores.append(recyclability * fraction)
        
        # Compute weighted averages
        avg_abundance = sum(abundance_scores)
        avg_toxicity = sum(toxicity_scores)
        avg_recyclability = sum(recyclability_scores)
        
        # Rarity index (focuses on the rarest element)
        rarest_element = min([self.default_sustainability.get(str(el), self.default_values)[0] 
                             for el in elements])
        
        # Overall sustainability score (higher is better)
        overall_score = avg_abundance * 0.4 + (1.0 - avg_toxicity) * 0.4 + avg_recyclability * 0.2
        
        return {
            'abundance_score': avg_abundance,
            'toxicity_score': avg_toxicity,
            'recyclability_score': avg_recyclability,
            'rarest_element_score': rarest_element,
            'overall_sustainability': overall_score
        }
    
    def enrich_graph(self, graph, structure):
        """
        Add sustainability features to a graph representation.
        
        Args:
            graph: torch_geometric.data.Data object
            structure: pymatgen Structure object
            
        Returns:
            Enriched graph with sustainability features
        """
        # Get sustainability features
        sust_features = self.get_features(structure)
        
        # Convert to tensor and add to graph
        sust_tensor = torch.tensor([
            sust_features['abundance_score'],
            sust_features['toxicity_score'],
            sust_features['recyclability_score']
        ], dtype=torch.float)
        
        # Add to graph as a global feature
        graph.sust_features = sust_tensor
        
        return graph
"""
Crystal Graph representation module for topological materials.

This module provides utilities for converting crystal structures into graph representations
that can be used with graph neural networks, specifically designed to capture features
relevant for topological material properties.
"""

import torch
import numpy as np
from pymatgen.core import Structure
from torch_geometric.data import Data


class CrystalGraphConverter:
    """
    Converts crystal structures to graph representations.
    
    This class handles the conversion of pymatgen Structure objects to
    graph representations suitable for machine learning, with special
    attention to features that influence topological properties.
    """
    
    def __init__(self, radius=6.0, max_neighbors=12, periodic=True,
                 atom_features=None, edge_features=None):
        """
        Initialize the crystal graph converter.
        
        Args:
            radius: Cutoff radius for neighbor search
            max_neighbors: Maximum number of neighbors per atom
            periodic: Whether to use periodic boundary conditions
            atom_features: Custom atom feature function (optional)
            edge_features: Custom edge feature function (optional)
        """
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.periodic = periodic
        self.atom_features = atom_features or self._default_atom_features
        self.edge_features = edge_features or self._default_edge_features
        
        # Precompute element properties for faster access
        self._setup_element_properties()
        
    def _setup_element_properties(self):
        """Precompute element properties for faster access."""
        # Atomic number -> electronegativity, atomic radius, etc.
        self.element_props = {
            # Format: Z: [electronegativity, atomic_radius, s_valence, p_valence, d_valence, f_valence]
            1: [2.20, 0.38, 1, 0, 0, 0],  # H
            2: [0.00, 0.32, 2, 0, 0, 0],  # He
            3: [0.98, 1.52, 1, 0, 0, 0],  # Li
            4: [1.57, 1.12, 2, 0, 0, 0],  # Be
            5: [2.04, 0.85, 2, 1, 0, 0],  # B
            6: [2.55, 0.75, 2, 2, 0, 0],  # C
            7: [3.04, 0.71, 2, 3, 0, 0],  # N
            8: [3.44, 0.63, 2, 4, 0, 0],  # O
            # ... Add more elements as needed
            # For the complete dataset you would include all elements
        }
        
        # Default values for missing elements
        self.default_props = [2.0, 1.0, 0, 0, 0, 0]
        
    def _default_atom_features(self, element):
        """
        Generate default atom features.
        
        Args:
            element: Element object from pymatgen
            
        Returns:
            Tensor of atom features
        """
        atomic_num = element.Z
        # Get element properties or use defaults
        props = self.element_props.get(atomic_num, self.default_props)
        
        features = [
            atomic_num,
            props[0],  # Electronegativity
            props[1],  # Atomic radius
            props[2],  # s valence
            props[3],  # p valence
            props[4],  # d valence
            props[5],  # f valence
            element.X,  # Electronegativity (Pauling)
            element.row,  # Period
            element.group,  # Group
            element.is_metal,  # Is metal
            element.is_transition_metal,  # Is transition metal
        ]
        
        return torch.tensor(features, dtype=torch.float)
    
    def _default_edge_features(self, dist, vector):
        """
        Generate default edge features.
        
        Args:
            dist: Distance between atoms
            vector: Vector between atoms
            
        Returns:
            Tensor of edge features
        """
        # Compute Gaussian basis functions
        gaussian_means = torch.linspace(0, self.radius, 20)
        gaussian_width = gaussian_means[1] - gaussian_means[0]
        
        # Distance features using Gaussian basis
        dist_tensor = torch.tensor([dist], dtype=torch.float)
        gaussian_features = torch.exp(-(dist_tensor - gaussian_means)**2 / (2 * gaussian_width**2))
        
        # Spherical harmonics features could be added here for better angular resolution
        
        return gaussian_features
    
    def convert(self, structure):
        """
        Convert a pymatgen Structure to a graph representation.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            torch_geometric.data.Data object
        """
        if not isinstance(structure, Structure):
            raise TypeError("Expected a pymatgen Structure object")
        
        # Get neighbors within cutoff radius
        num_atoms = len(structure)
        all_neighbors = structure.get_all_neighbors(self.radius, include_index=True)
        
        # Initialize node features
        node_features = []
        for site in structure.sites:
            node_features.append(self.atom_features(site.specie))
        
        # Stack node features
        if node_features:
            node_features = torch.stack(node_features)
        else:
            # Fallback if no atoms (should never happen)
            node_features = torch.zeros((0, 12), dtype=torch.float)
            
        # Get node positions (fractional coordinates)
        pos_frac = torch.tensor(structure.frac_coords, dtype=torch.float)
        # Get node positions (cartesian coordinates)
        pos_cart = torch.tensor(structure.cart_coords, dtype=torch.float)
        
        # Extract lattice information
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float)
        
        # Process edges (limit to max_neighbors per atom if needed)
        edge_indices = []
        edge_features = []
        edge_vectors = []
        
        for i, neighbors in enumerate(all_neighbors):
            # Sort neighbors by distance
            neighbors = sorted(neighbors, key=lambda x: x[1])
            
            # Limit number of neighbors if needed
            if self.max_neighbors > 0:
                neighbors = neighbors[:self.max_neighbors]
                
            for neighbor, dist, j in neighbors:
                edge_indices.append((i, j))
                
                # Get edge vector (respecting periodic boundary conditions)
                vector = structure.sites[j].coords - structure.sites[i].coords
                if self.periodic:
                    vector = structure.lattice.get_distance_and_image(
                        structure.sites[i].frac_coords,
                        structure.sites[j].frac_coords)[1]
                    vector = np.dot(vector, structure.lattice.matrix)
                
                edge_vectors.append(vector)
                edge_features.append(self.edge_features(dist, vector))
        
        # Convert to tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_features) if edge_features else None
            edge_vec = torch.tensor(edge_vectors, dtype=torch.float) if edge_vectors else None
        else:
            # Fallback for isolated atoms
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = None
            edge_vec = None
        
        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos_cart,
            frac_coords=pos_frac,
            lattice=lattice,
            edge_vec=edge_vec,
            num_nodes=num_atoms
        )
        
        return data
    
    def convert_batch(self, structures):
        """
        Convert a batch of structures to graph representations.
        
        Args:
            structures: List of pymatgen Structure objects
            
        Returns:
            List of torch_geometric.data.Data objects
        """
        return [self.convert(structure) for structure in structures]


class TopologicalFeatures:
    """
    Extract topological feature descriptors from crystal structures.
    
    This class adds additional features specifically relevant for
    topological materials prediction.
    """
    
    def __init__(self):
        """Initialize the topological features extractor."""
        # Elements often found in topological materials
        self.topo_elements = ['Bi', 'Sb', 'Te', 'Se', 'Sn', 'Pb', 'Hg', 'Tl']
        
        # Space groups that often host topological phases
        self.topo_spacegroups = [2, 164, 166, 176, 187, 189, 191, 194, 221, 224, 229]
    
    def get_features(self, structure):
        """
        Extract topological features from a structure.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Dictionary of topological features
        """
        # Calculate composition-based features
        elements = [str(el) for el in structure.composition.elements]
        num_topo_elements = sum(1 for el in elements if el in self.topo_elements)
        frac_topo_elements = num_topo_elements / len(elements) if elements else 0
        
        # Space group information
        sg_num = structure.get_space_group_info()[1]
        is_topo_spacegroup = sg_num in self.topo_spacegroups
        
        # Inversion symmetry
        has_inversion = structure.has_inversion_symmetry()
        
        # Band theory relevant descriptors
        avg_electronegativity = np.mean([el.X for el in structure.composition.elements])
        electronegativity_diff = np.max([el.X for el in structure.composition.elements]) - \
                                  np.min([el.X for el in structure.composition.elements])
        
        # Return features as a dictionary
        return {
            'num_topo_elements': num_topo_elements,
            'frac_topo_elements': frac_topo_elements,
            'is_topo_spacegroup': is_topo_spacegroup,
            'has_inversion': has_inversion,
            'avg_electronegativity': avg_electronegativity,
            'electronegativity_diff': electronegativity_diff,
            'space_group_num': sg_num
        }
    
    def enrich_graph(self, graph, structure):
        """
        Add topological features to a graph representation.
        
        Args:
            graph: torch_geometric.data.Data object
            structure: pymatgen Structure object
            
        Returns:
            Enriched graph with topological features
        """
        # Get topological features
        topo_features = self.get_features(structure)
        
        # Convert to tensor and add to graph
        topo_tensor = torch.tensor([
            topo_features['num_topo_elements'],
            topo_features['frac_topo_elements'],
            float(topo_features['is_topo_spacegroup']),
            float(topo_features['has_inversion']),
            topo_features['avg_electronegativity'],
            topo_features['electronegativity_diff'],
            topo_features['space_group_num'] / 230.0  # Normalize
        ], dtype=torch.float)
        
        # Add to graph as a global feature
        graph.topo_features = topo_tensor
        
        return graph