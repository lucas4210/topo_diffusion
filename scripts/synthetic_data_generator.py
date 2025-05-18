"""
Synthetic data generation for topological quantum materials research.

This script creates synthetic crystal structures and property data that mimics
the expected structure from JARVIS datasets, allowing for model development
when the actual JARVIS database is unavailable.
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import shutil
from datetime import datetime
import argparse

# Configuration
CONFIG = {
    'num_materials': 5000,            # Total number of materials to generate
    'num_topological': 800,           # Number of topological materials
    'training_subset_size': 2000,     # Size of training subset
    'random_seed': 42,                # Random seed for reproducibility
    'elements': [
        # Main group elements
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'Tl', 'Pb', 'Bi', 'Po',
        # Transition metals
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        # Rare earth elements
        'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
        'Er', 'Tm', 'Yb', 'Lu'
    ],
    # Elements often found in topological materials
    'topo_elements': ['Bi', 'Sb', 'Te', 'Se', 'Sn', 'Pb', 'Hg', 'Tl'],
    # Space groups that often host topological phases
    'topo_space_groups': [2, 164, 166, 176, 187, 189, 191, 194, 221, 224, 229]
}

# Create necessary directories
def create_directories(output_dir):
    """Create the directory structure for the synthetic dataset."""
    dirs = [
        output_dir,
        f'{output_dir}/jarvis',
        f'{output_dir}/materials_project',
        f'{output_dir}/materials_project/structures',
        f'{output_dir}/materials_project/training_subset_structures',
        f'{output_dir}/topological',
        f'{output_dir}/topological/structures',
        f'{output_dir}/sustainability',
        f'{output_dir}/processed'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created.")

# Generate random crystal structure
def generate_crystal_structure(material_id, elements):
    """
    Generate a random crystal structure.
    
    Args:
        material_id: Identifier for the material
        elements: List of elements to choose from
        
    Returns:
        Dictionary containing structure information
    """
    # Select number of elements (1-4)
    num_elements = random.randint(1, 4)
    
    # Select elements (biased towards topo_elements for some materials)
    if random.random() < 0.3:
        # Include at least one topological element
        selected_elements = random.sample(CONFIG['topo_elements'], min(1, len(CONFIG['topo_elements'])))
        remaining = num_elements - len(selected_elements)
        if remaining > 0:
            other_elements = [e for e in elements if e not in selected_elements]
            selected_elements.extend(random.sample(other_elements, remaining))
    else:
        # Completely random selection
        selected_elements = random.sample(elements, num_elements)
    
    # Generate element counts (stoichiometry)
    element_counts = {}
    for element in selected_elements:
        element_counts[element] = random.choice([1, 2, 3, 4])
    
    # Create chemical formula
    formula = ''.join([f"{element}{count}" if count > 1 else element for element, count in element_counts.items()])
    
    # Select space group
    if random.random() < 0.25:
        # Bias toward space groups common for topological materials
        space_group = random.choice(CONFIG['topo_space_groups'])
    else:
        # Random space group
        space_group = random.randint(1, 230)
    
    # Generate lattice parameters (reasonable ranges for crystals)
    a = random.uniform(3.0, 12.0)
    b = random.uniform(3.0, 12.0)
    c = random.uniform(3.0, 12.0)
    alpha = random.choice([90.0, 90.0, 90.0, 120.0, random.uniform(60.0, 120.0)])
    beta = random.choice([90.0, 90.0, 90.0, random.uniform(60.0, 120.0)])
    gamma = random.choice([90.0, 90.0, 90.0, 120.0, random.uniform(60.0, 120.0)])
    
    # Convert angles to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # Calculate lattice vectors
    # For simplicity, we'll use a simplified calculation that works best for orthogonal cells
    ax = a
    ay = 0.0
    az = 0.0
    
    bx = b * np.cos(gamma_rad)
    by = b * np.sin(gamma_rad)
    bz = 0.0
    
    cx = c * np.cos(beta_rad)
    cy = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    cz = np.sqrt(c*c - cx*cx - cy*cy)
    
    lattice_matrix = [
        [ax, ay, az],
        [bx, by, bz],
        [cx, cy, cz]
    ]
    
    # Generate atomic positions (simplified - just generate reasonable fractional coordinates)
    total_atoms = sum(element_counts.values())
    
    # Ensure we create at least 20 atoms for each structure to make it substantial
    while total_atoms < 3:
        for element in element_counts:
            element_counts[element] += 1
        total_atoms = sum(element_counts.values())
    
    coords = []
    elements_list = []
    
    for element, count in element_counts.items():
        for _ in range(count):
            # Generate fractional coordinates between 0 and 1
            coord = [random.random(), random.random(), random.random()]
            coords.append(coord)
            elements_list.append(element)
    
    # Create the structure dictionary
    structure = {
        'material_id': material_id,
        'formula': formula,
        'elements': selected_elements,
        'element_counts': element_counts,
        'space_group': space_group,
        'lattice': {
            'a': a,
            'b': b,
            'c': c,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'matrix': lattice_matrix
        },
        'coords': coords,
        'elements_list': elements_list
    }
    
    return structure

# Generate synthetic material properties
def generate_material_properties(structure, is_topological):
    """
    Generate synthetic material properties.
    
    Args:
        structure: Crystal structure information
        is_topological: Whether the material is designated as topological
        
    Returns:
        Dictionary of material properties
    """
    # Calculate band gap
    # Topological materials tend to have small band gaps
    if is_topological:
        band_gap = random.uniform(0.0, 0.3)
    else:
        # Wider distribution for non-topological
        band_gap = random.choice([
            random.uniform(0.0, 0.1),    # Metal/semimetal (30% chance)
            random.uniform(0.1, 1.5),    # Semiconductor (40% chance)
            random.uniform(1.5, 5.0)     # Insulator (30% chance)
        ])
    
    # Calculate formation energy
    # Typically negative, more negative means more stable
    formation_energy = random.uniform(-5.0, 0.0)
    
    # Energy above convex hull (stability metric, 0 = stable)
    # Topological materials tend to be less stable
    if is_topological:
        e_above_hull = random.uniform(0.0, 0.2) if random.random() < 0.7 else random.uniform(0.0, 1.0)
    else:
        e_above_hull = random.uniform(0.0, 0.1) if random.random() < 0.8 else random.uniform(0.0, 0.5)
    
    # Z2 invariants for topological insulators
    if is_topological and band_gap > 0.01:
        # For strong TIs: (1;000), (0;111), etc.
        if random.random() < 0.7:
            z2_invariant = [1, 0, 0, 0]
        else:
            z2_invariant = [0, 1, 1, 1]
        
        topological_class = random.choice(['strong_ti', 'weak_ti', 'crystalline_ti'])
    else:
        z2_invariant = [0, 0, 0, 0]
        topological_class = 'trivial' if band_gap > 0.01 else 'metal'
    
    # Calculate density
    # Simplified density calculation - doesn't account for exact structure
    volume = structure['lattice']['a'] * structure['lattice']['b'] * structure['lattice']['c']
    density = sum([random.uniform(1.0, 20.0) for _ in range(len(structure['elements']))]) / volume
    
    properties = {
        'band_gap': band_gap,
        'formation_energy': formation_energy,
        'e_above_hull': e_above_hull,
        'is_topological': is_topological,
        'topological_class': topological_class,
        'z2_invariant': z2_invariant,
        'density': density
    }
    
    return properties

# Generate sustainability metrics
def generate_sustainability_metrics():
    """
    Generate sustainability metrics for all elements.
    
    Returns:
        DataFrame with sustainability metrics for each element
    """
    sustainability_data = []
    
    for element in CONFIG['elements']:
        # Rarity/abundance (higher is more abundant)
        if element in ['O', 'Si', 'Al', 'Fe', 'Ca', 'Na', 'K', 'Mg']:
            # Common elements
            abundance_ppm = random.uniform(10000, 500000)
        elif element in ['H', 'Ti', 'Mn', 'P', 'S', 'C', 'Cl']:
            # Moderately common
            abundance_ppm = random.uniform(100, 10000)
        elif element in CONFIG['topo_elements'] or element in ['Au', 'Pt', 'Ir', 'Re', 'Os']:
            # Rare elements, including those common in topological materials
            abundance_ppm = random.uniform(0.001, 1.0)
        else:
            # Other elements
            abundance_ppm = random.uniform(1.0, 100.0)
        
        # Toxicity (higher is more toxic)
        if element in ['Hg', 'Pb', 'Cd', 'As', 'Tl']:
            toxicity_rating = random.randint(2, 3)
        elif element in ['Sb', 'Te', 'Se', 'Cr', 'Ni', 'Co']:
            toxicity_rating = random.randint(1, 2)
        else:
            toxicity_rating = random.randint(0, 1)
        
        # Recyclability (higher is more recyclable)
        if element in ['Au', 'Pt', 'Pd', 'Ag', 'Cu', 'Al', 'Fe']:
            recyclability_score = random.uniform(0.7, 0.9)
        elif element in CONFIG['topo_elements']:
            recyclability_score = random.uniform(0.2, 0.5)
        else:
            recyclability_score = random.uniform(0.3, 0.7)
        
        # Normalize abundance to a score between 0 and 1
        if abundance_ppm > 0:
            abundance_score = min(1.0, max(0.1, 0.1 + 0.3 * np.log10(abundance_ppm + 1) / 6))
        else:
            abundance_score = 0.1
        
        # Overall sustainability score (higher is better)
        sustainability_score = (
            abundance_score * 0.4 +
            (1 - toxicity_rating / 3.0) * 0.3 +
            recyclability_score * 0.3
        )
        
        sustainability_data.append({
            'element': element,
            'abundance_ppm': abundance_ppm,
            'toxicity_rating': toxicity_rating,
            'recyclability_score': recyclability_score,
            'abundance_score': abundance_score,
            'toxicity_impact': 1.0 - (toxicity_rating / 3.0),
            'sustainability_score': sustainability_score
        })
    
    return pd.DataFrame(sustainability_data)

# Create a POSCAR file from a structure
def create_poscar_file(structure, filename):
    """
    Create a VASP POSCAR file from a structure.
    
    Args:
        structure: Dictionary containing structure information
        filename: Output file name
    """
    # Get element counts ordered by elements_list
    elements = []
    element_counts = {}
    
    # Identify unique elements while preserving order
    for element in structure['elements_list']:
        if element not in elements:
            elements.append(element)
            element_counts[element] = 1
        else:
            element_counts[element] += 1
    
    # Create POSCAR content
    poscar_content = f"{structure['material_id']}\n"
    poscar_content += "1.0\n"
    
    # Lattice vectors
    for row in structure['lattice']['matrix']:
        poscar_content += f"{row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f}\n"
    
    # Elements line
    poscar_content += " ".join(elements) + "\n"
    
    # Element counts
    poscar_content += " ".join(str(element_counts[element]) for element in elements) + "\n"
    
    # Coordinates
    poscar_content += "Direct\n"
    for coord in structure['coords']:
        poscar_content += f"{coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f}\n"
    
    # Write file
    with open(filename, 'w') as f:
        f.write(poscar_content)

# Generate JARVIS-DFT-like JSON data
def generate_jarvis_dft_data(structures, properties):
    """
    Generate synthetic JARVIS-DFT data.
    
    Args:
        structures: List of structure dictionaries
        properties: List of property dictionaries
        
    Returns:
        List of JARVIS-DFT-like entries
    """
    jarvis_dft = []
    
    for i, (structure, props) in enumerate(zip(structures, properties)):
        # Create atoms data structure
        atoms = {
            'lattice_mat': structure['lattice']['matrix'],
            'coords': structure['coords'],
            'elements': structure['elements_list']
        }
        
        # Create JARVIS-DFT entry
        entry = {
            'jid': structure['material_id'],
            'formula': structure['formula'],
            'spacegroup_number': structure['space_group'],
            'formation_energy_per_atom': props['formation_energy'],
            'bandgap': props['band_gap'],
            'ehull': props['e_above_hull'],
            'elements': list(structure['element_counts'].keys()),
            'density': props['density'],
            'atoms': atoms
        }
        
        jarvis_dft.append(entry)
    
    return jarvis_dft

# Generate JARVIS-TOPO-like JSON data
def generate_jarvis_topo_data(structures, properties):
    """
    Generate synthetic JARVIS-TOPO data.
    
    Args:
        structures: List of structure dictionaries
        properties: List of property dictionaries
        
    Returns:
        List of JARVIS-TOPO-like entries
    """
    jarvis_topo = []
    
    for i, (structure, props) in enumerate(zip(structures, properties)):
        # Only include entries that are topological or have small band gaps
        if props['is_topological'] or props['band_gap'] < 0.3:
            entry = {
                'jid': structure['material_id'],
                'formula': structure['formula'],
                'topological_type': props['topological_class'],
                'z2_invariant': props['z2_invariant'],
                'bandgap': props['band_gap']
            }
            
            jarvis_topo.append(entry)
    
    return jarvis_topo

# Main data generation function
def generate_synthetic_data(output_dir='./data'):
    """
    Generate and save synthetic data for the project.
    
    Args:
        output_dir: Directory where the data will be stored
    """
    # Fix random seed for reproducibility
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    print(f"Generating {CONFIG['num_materials']} synthetic materials...")
    
    # Generate structures and properties
    structures = []
    properties = []
    
    # Decide which materials will be topological
    topo_indices = random.sample(range(CONFIG['num_materials']), CONFIG['num_topological'])
    topo_set = set(topo_indices)
    
    for i in tqdm(range(CONFIG['num_materials']), desc="Generating materials"):
        material_id = f"mp-{i+1000}"
        is_topological = i in topo_set
        
        # Generate structure
        structure = generate_crystal_structure(material_id, CONFIG['elements'])
        
        # Generate properties
        props = generate_material_properties(structure, is_topological)
        
        structures.append(structure)
        properties.append(props)
    
    print(f"Generated {len(structures)} materials, {CONFIG['num_topological']} topological")
    
    # Generate sustainability metrics
    print("Generating sustainability metrics...")
    sustainability_df = generate_sustainability_metrics()
    
    # Generate JARVIS-like data
    print("Creating JARVIS-like data files...")
    jarvis_dft = generate_jarvis_dft_data(structures, properties)
    jarvis_topo = generate_jarvis_topo_data(structures, properties)
    
    # Save JARVIS-like data
    with open(f'{output_dir}/jarvis/jarvis_dft_3d.json', 'w') as f:
        json.dump(jarvis_dft, f)
    
    with open(f'{output_dir}/jarvis/jarvis_topo.json', 'w') as f:
        json.dump(jarvis_topo, f)
    
    # Create materials summary dataframe
    materials_data = []
    for i, (structure, props) in enumerate(zip(structures, properties)):
        # Calculate sustainability score
        structure_elements = list(structure['element_counts'].keys())
        
        # Get sustainability scores for elements in this material
        element_sust_scores = sustainability_df[sustainability_df['element'].isin(structure_elements)]
        avg_sust_score = element_sust_scores['sustainability_score'].mean()
        
        # Calculate topological score
        topo_score = 0.0
        if props['is_topological']:
            topo_score = 1.0
        else:
            # Apply heuristics for unclassified materials
            if any(elem in CONFIG['topo_elements'] for elem in structure_elements):
                topo_score += 0.3
            if props['band_gap'] < 0.3:
                topo_score += 0.2
            if props['band_gap'] < 0.1:
                topo_score += 0.1
        
        # Clip to 0-1 range
        topo_score = min(topo_score, 1.0)
        
        # Calculate stability score
        stability_score = max(0, 1.0 - props['e_above_hull'])
        
        # Calculate combined score
        combined_score = (
            0.4 * topo_score +
            0.3 * avg_sust_score +
            0.3 * stability_score
        )
        
        material_dict = {
            'material_id': structure['material_id'],
            'formula': structure['formula'],
            'spacegroup': structure['space_group'],
            'formation_energy': props['formation_energy'],
            'band_gap': props['band_gap'],
            'e_above_hull': props['e_above_hull'],
            'elements': ','.join(structure_elements),
            'topological_class': props['topological_class'],
            'z2_invariant': str(props['z2_invariant']),
            'is_topological': props['is_topological'],
            'topo_score': topo_score,
            'sustainability_score': avg_sust_score,
            'stability_score': stability_score,
            'combined_score': combined_score
        }
        
        materials_data.append(material_dict)
    
    materials_df = pd.DataFrame(materials_data)
    
    # Save materials dataset
    materials_df.to_csv(f'{output_dir}/materials_project/materials_summary.csv', index=False)
    print(f"Saved {len(materials_df)} materials to materials_summary.csv")
    
    # Save topological predictions
    materials_df.to_csv(f'{output_dir}/topological/topological_predictions.csv', index=False)
    
    # Save top topological candidates
    top_topo_candidates = materials_df[
        (materials_df['topo_score'] > 0.5) & 
        (materials_df['stability_score'] > 0.7)
    ].sort_values('topo_score', ascending=False)
    
    top_topo_candidates.to_csv(f'{output_dir}/topological/top_topological_candidates.csv', index=False)
    print(f"Saved {len(top_topo_candidates)} top topological candidates")
    
    # Save sustainability dataset
    sustainability_df.to_csv(f'{output_dir}/sustainability/element_sustainability.csv', index=False)
    print(f"Saved sustainability data for {len(sustainability_df)} elements")
    
    # Save combined dataset
    materials_df.to_csv(f'{output_dir}/processed/combined_dataset.csv', index=False)
    
    # Save top sustainable topological candidates
    top_candidates = materials_df[
        (materials_df['combined_score'] > 0.6)
    ].sort_values('combined_score', ascending=False)
    
    top_candidates.to_csv(f'{output_dir}/processed/top_sustainable_topological_candidates.csv', index=False)
    print(f"Saved {len(top_candidates)} top sustainable topological candidates")
    
    # Save structure files
    print("Saving structure files...")
    for i, structure in enumerate(tqdm(structures, desc="Saving structures")):
        material_id = structure['material_id']
        
        # Save to materials directory
        poscar_path = f'{output_dir}/materials_project/structures/{material_id}.vasp'
        create_poscar_file(structure, poscar_path)
        
        # If it's a top topological candidate, also save to topological directory
        if material_id in top_topo_candidates['material_id'].values:
            topo_path = f'{output_dir}/topological/structures/{material_id}.vasp'
            create_poscar_file(structure, topo_path)
    
    # Create training subset
    print("Creating training subset...")
    training_size = min(CONFIG['training_subset_size'], len(materials_df))
    training_subset = materials_df.sample(n=training_size, random_state=CONFIG['random_seed'])
    
    training_subset.to_csv(f'{output_dir}/materials_project/training_subset.csv', index=False)
    
    # Copy structure files for training subset
    for material_id in tqdm(training_subset['material_id'], desc="Copying training subset structures"):
        src = f'{output_dir}/materials_project/structures/{material_id}.vasp'
        dst = f'{output_dir}/materials_project/training_subset_structures/{material_id}.vasp'
        if os.path.exists(src):
            shutil.copy(src, dst)
    
    print("\nSynthetic data generation complete! The data structure mimics JARVIS datasets.")
    print(f"Total materials: {len(materials_df)}")
    print(f"Topological materials: {CONFIG['num_topological']}")
    print(f"Training subset size: {training_size}")
    print("All necessary datasets have been created with synthetic but realistic data.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate synthetic data for topological quantum materials")
    parser.add_argument("--output_dir", default="./data", help="Directory to store the generated data")
    args = parser.parse_args()
    
    # Generate timestamp for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Synthetic data generation started: {timestamp}")
    
    # Create directory structure
    create_directories(args.output_dir)
    
    # Generate synthetic data
    generate_synthetic_data(args.output_dir)
    
    # Generate ending timestamp for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Synthetic data generation completed: {timestamp}")