import json
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import shutil

# Configuration
TRAINING_SUBSET_SIZE = 5000  # Adjust this value as needed

# Create necessary directories
def create_directories():
    os.makedirs('data/jarvis', exist_ok=True)
    os.makedirs('data/materials_project', exist_ok=True)
    os.makedirs('data/topological', exist_ok=True)
    os.makedirs('data/sustainability', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/materials_project/structures', exist_ok=True)
    os.makedirs('data/topological/structures', exist_ok=True)
    os.makedirs('data/materials_project/training_subset_structures', exist_ok=True)

def main():
    # Create directories
    create_directories()
    
    # Load JARVIS-DFT data (materials structures and properties)
    print("Loading JARVIS-DFT data...")
    with open('data/jarvis/jarvis_dft_3d.json', 'r') as f:
        jarvis_dft = json.load(f)

    # Load JARVIS-TOPO data (topological classifications)
    print("Loading JARVIS-TOPO data...")
    with open('data/jarvis/jarvis_topo.json', 'r') as f:
        jarvis_topo = json.load(f)

    # Create mapping of JARVIS IDs to topological properties
    topo_mapping = {}
    for material in jarvis_topo:
        jid = material.get('jid', '')
        if jid:
            topo_mapping[jid] = {
                'topological_class': material.get('topological_type', ''),
                'z2_invariant': material.get('z2_invariant', []),
                'band_gap': material.get('bandgap', None)
            }

    # Process JARVIS-DFT materials
    materials_data = []
    for material in tqdm(jarvis_dft, desc="Processing materials"):
        jid = material.get('jid', '')
        
        # Get basic material properties
        material_dict = {
            'material_id': jid,
            'formula': material.get('formula', ''),
            'spacegroup': material.get('spacegroup_number', ''),
            'formation_energy': material.get('formation_energy_per_atom', 0.0),
            'band_gap': material.get('bandgap', 0.0),
            'e_above_hull': material.get('ehull', 0.0) if 'ehull' in material else 0.0,
            'elements': ','.join(material.get('elements', [])),
        }
        
        # Add topological properties if available
        if jid in topo_mapping:
            material_dict['topological_class'] = topo_mapping[jid]['topological_class']
            material_dict['z2_invariant'] = str(topo_mapping[jid]['z2_invariant'])
            material_dict['is_topological'] = topo_mapping[jid]['topological_class'] != 'trivial'
        else:
            material_dict['topological_class'] = None
            material_dict['z2_invariant'] = None
            material_dict['is_topological'] = False
        
        # Calculate topological score
        topo_score = 0.0
        if material_dict['is_topological']:
            topo_score = 1.0  # Known topological materials get highest score
        else:
            # Apply heuristics for unclassified materials
            topo_elements = ['Bi', 'Sb', 'Te', 'Se', 'Sn', 'Pb', 'Hg', 'Tl']
            if any(elem in material_dict['elements'].split(',') for elem in topo_elements):
                topo_score += 0.3
            if material_dict['band_gap'] is not None:
                if material_dict['band_gap'] < 0.3:
                    topo_score += 0.2
                if material_dict['band_gap'] < 0.1:
                    topo_score += 0.1
        
        material_dict['topo_score'] = min(topo_score, 1.0)  # Clip to 0-1 range
        
        # Calculate sustainability score based on elements
        # This is a simplified approach - you would ideally use real sustainability data
        element_sustainability = {
            'H': 0.9, 'He': 0.1, 'Li': 0.4, 'Be': 0.1, 'B': 0.5, 'C': 0.9, 'N': 0.9, 'O': 0.9, 'F': 0.5, 'Ne': 0.1,
            'Na': 0.8, 'Mg': 0.8, 'Al': 0.8, 'Si': 0.9, 'P': 0.6, 'S': 0.8, 'Cl': 0.6, 'Ar': 0.1, 'K': 0.7, 'Ca': 0.8,
            'Sc': 0.3, 'Ti': 0.7, 'V': 0.5, 'Cr': 0.5, 'Mn': 0.6, 'Fe': 0.8, 'Co': 0.3, 'Ni': 0.4, 'Cu': 0.5, 'Zn': 0.6,
            'Ga': 0.3, 'Ge': 0.3, 'As': 0.2, 'Se': 0.3, 'Br': 0.4, 'Kr': 0.1, 'Rb': 0.3, 'Sr': 0.5, 'Y': 0.3, 'Zr': 0.5,
            'Nb': 0.3, 'Mo': 0.4, 'Tc': 0.1, 'Ru': 0.2, 'Rh': 0.1, 'Pd': 0.1, 'Ag': 0.3, 'Cd': 0.2, 'In': 0.2, 'Sn': 0.4,
            'Sb': 0.3, 'Te': 0.2, 'I': 0.4, 'Xe': 0.1, 'Cs': 0.2, 'Ba': 0.5, 'La': 0.3, 'Ce': 0.3, 'Pr': 0.2, 'Nd': 0.2,
            'Pm': 0.1, 'Sm': 0.2, 'Eu': 0.1, 'Gd': 0.2, 'Tb': 0.1, 'Dy': 0.2, 'Ho': 0.2, 'Er': 0.2, 'Tm': 0.1, 'Yb': 0.2,
            'Lu': 0.2, 'Hf': 0.3, 'Ta': 0.2, 'W': 0.3, 'Re': 0.1, 'Os': 0.1, 'Ir': 0.1, 'Pt': 0.1, 'Au': 0.2, 'Hg': 0.1,
            'Tl': 0.1, 'Pb': 0.3, 'Bi': 0.3, 'Po': 0.1, 'At': 0.1, 'Rn': 0.1, 'Fr': 0.1, 'Ra': 0.1, 'Ac': 0.1, 'Th': 0.1,
            'Pa': 0.1, 'U': 0.1, 'Np': 0.1, 'Pu': 0.1
        }
        
        elements = material_dict['elements'].split(',')
        sust_scores = [element_sustainability.get(elem, 0.5) for elem in elements]
        material_dict['sustainability_score'] = np.mean(sust_scores)
        
        # Calculate stability score (inverse of energy above hull)
        if material_dict['e_above_hull'] is not None:
            material_dict['stability_score'] = max(0, 1.0 - material_dict['e_above_hull'])
        else:
            material_dict['stability_score'] = 0.5  # Default value
        
        # Calculate combined score with weights
        weights = {
            'topological': 0.4,
            'sustainability': 0.3,
            'stability': 0.3
        }
        
        material_dict['combined_score'] = (
            weights['topological'] * material_dict['topo_score'] +
            weights['sustainability'] * material_dict['sustainability_score'] +
            weights['stability'] * material_dict['stability_score']
        )
        
        materials_data.append(material_dict)

    # Convert to DataFrame
    materials_df = pd.DataFrame(materials_data)

    # Save materials dataset
    materials_df.to_csv('data/materials_project/materials_summary.csv', index=False)
    print(f"Saved {len(materials_df)} materials to materials_summary.csv")

    # Save topological predictions
    materials_df.to_csv('data/topological/topological_predictions.csv', index=False)

    # Save top topological candidates
    top_topo_candidates = materials_df[
        (materials_df['topo_score'] > 0.5) & 
        (materials_df['stability_score'] > 0.7)  # Stable materials
    ].sort_values('topo_score', ascending=False)

    top_topo_candidates.to_csv('data/topological/top_topological_candidates.csv', index=False)
    print(f"Saved {len(top_topo_candidates)} top topological candidates")

    # Save sustainability dataset
    element_data = []
    for element, score in element_sustainability.items():
        # Convert sustainability score to abundance and toxicity
        abundance = score * 0.8 + 0.1  # Scale from 0.1-0.9
        toxicity = 1.0 - score         # Inverse of sustainability
        
        element_data.append({
            'element': element,
            'abundance_score': abundance,
            'toxicity_score': toxicity,
            'sustainability_score': score
        })

    sustainability_df = pd.DataFrame(element_data)
    sustainability_df.to_csv('data/sustainability/element_sustainability.csv', index=False)
    print(f"Saved sustainability data for {len(sustainability_df)} elements")

    # Save combined dataset
    materials_df.to_csv('data/processed/combined_dataset.csv', index=False)

    # Save top sustainable topological candidates
    top_candidates = materials_df[
        (materials_df['combined_score'] > 0.6)
    ].sort_values('combined_score', ascending=False)

    top_candidates.to_csv('data/processed/top_sustainable_topological_candidates.csv', index=False)
    print(f"Saved {len(top_candidates)} top sustainable topological candidates")

    # Save structure files
    print("Processing structure files...")
    # Process structure files
    for material in tqdm(jarvis_dft, desc="Saving structures"):
        jid = material.get('jid', '')
        if not jid:
            continue
            
        # Get structure data
        if 'atoms' in material:
            # Create POSCAR format data
            atoms_data = material['atoms']
            lattice = atoms_data.get('lattice_mat', [])
            coords = atoms_data.get('coords', [])
            elements = atoms_data.get('elements', [])
            
            if lattice and coords and elements:
                # Create POSCAR content
                poscar_content = f"{jid}\n"
                poscar_content += "1.0\n"
                for row in lattice:
                    poscar_content += f"{row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f}\n"
                
                # Elements line
                unique_elements = []
                element_counts = {}
                for elem in elements:
                    if elem not in element_counts:
                        unique_elements.append(elem)
                        element_counts[elem] = 1
                    else:
                        element_counts[elem] += 1
                        
                poscar_content += " ".join(unique_elements) + "\n"
                poscar_content += " ".join(str(element_counts[elem]) for elem in unique_elements) + "\n"
                
                # Coordinates
                poscar_content += "Direct\n"
                for coord in coords:
                    poscar_content += f"{coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f}\n"
                
                # Save to materials directory
                with open(f'data/materials_project/structures/{jid}.vasp', 'w') as f:
                    f.write(poscar_content)
                
                # If it's a top topological candidate, also save to topological directory
                if jid in top_topo_candidates['material_id'].values:
                    with open(f'data/topological/structures/{jid}.vasp', 'w') as f:
                        f.write(poscar_content)

    # Create training subset
    print("Creating training subset...")
    training_size = min(TRAINING_SUBSET_SIZE, len(materials_df))
    training_subset = materials_df.sample(n=training_size, random_state=42)

    training_subset.to_csv('data/materials_project/training_subset.csv', index=False)

    # Copy structure files for training subset
    for material_id in tqdm(training_subset['material_id'], desc="Copying training subset structures"):
        src = f'data/materials_project/structures/{material_id}.vasp'
        dst = f'data/materials_project/training_subset_structures/{material_id}.vasp'
        if os.path.exists(src):
            shutil.copy(src, dst)

    # Create a more detailed sustainability dataset
    create_sustainability_dataset()

    print("Data processing complete! All necessary datasets have been created from JARVIS data.")

def create_sustainability_dataset():
    """Create a more detailed sustainability dataset with real-world data"""
    print("Creating detailed sustainability dataset...")
    
    # Data from USGS and environmental databases
    # Format: element: [abundance_ppm, toxicity_rating, recyclability_score]
    sustainability_data = {
        'H': [1400, 0, 0.9],
        'He': [0.008, 0, 0.1],
        'Li': [20, 1, 0.4],
        'Be': [2.8, 3, 0.2],
        'B': [10, 1, 0.5],
        'C': [200, 0, 0.7],
        'N': [20, 0, 0.8],
        'O': [461000, 0, 0.9],
        'F': [585, 2, 0.4],
        'Ne': [0.005, 0, 0.1],
        'Na': [23600, 1, 0.8],
        'Mg': [23300, 0, 0.8],
        'Al': [82300, 0, 0.8],
        'Si': [282000, 0, 0.9],
        'P': [1050, 1, 0.6],
        'S': [350, 1, 0.7],
        'Cl': [145, 2, 0.6],
        'Ar': [3.5, 0, 0.1],
        'K': [20900, 0, 0.7],
        'Ca': [41500, 0, 0.8],
        'Sc': [22, 0, 0.3],
        'Ti': [5650, 0, 0.7],
        'V': [120, 1, 0.5],
        'Cr': [102, 2, 0.5],
        'Mn': [950, 1, 0.6],
        'Fe': [56300, 0, 0.8],
        'Co': [25, 1, 0.3],
        'Ni': [84, 1, 0.4],
        'Cu': [60, 1, 0.7],
        'Zn': [70, 1, 0.6],
        'Ga': [19, 1, 0.3],
        'Ge': [1.5, 1, 0.3],
        'As': [1.8, 3, 0.2],
        'Se': [0.05, 2, 0.3],
        'Br': [2.4, 2, 0.4],
        'Kr': [0.0001, 0, 0.1],
        'Rb': [90, 1, 0.3],
        'Sr': [370, 1, 0.5],
        'Y': [33, 0, 0.3],
        'Zr': [165, 0, 0.5],
        'Nb': [20, 0, 0.3],
        'Mo': [1.2, 1, 0.4],
        'Tc': [0, 2, 0.1],
        'Ru': [0.001, 1, 0.2],
        'Rh': [0.001, 1, 0.1],
        'Pd': [0.015, 1, 0.1],
        'Ag': [0.075, 1, 0.3],
        'Cd': [0.15, 3, 0.2],
        'In': [0.25, 1, 0.2],
        'Sn': [2.3, 1, 0.4],
        'Sb': [0.2, 2, 0.3],
        'Te': [0.001, 2, 0.2],
        'I': [0.45, 1, 0.4],
        'Xe': [0.00003, 0, 0.1],
        'Cs': [3, 1, 0.2],
        'Ba': [425, 1, 0.5],
        'La': [39, 0, 0.3],
        'Ce': [66.5, 0, 0.3],
        'Pr': [9.2, 0, 0.2],
        'Nd': [41.5, 0, 0.2],
        'Pm': [0, 1, 0.1],
        'Sm': [7.05, 0, 0.2],
        'Eu': [2, 0, 0.1],
        'Gd': [6.2, 0, 0.2],
        'Tb': [1.2, 0, 0.1],
        'Dy': [5.2, 0, 0.2],
        'Ho': [1.3, 0, 0.2],
        'Er': [3.5, 0, 0.2],
        'Tm': [0.52, 0, 0.1],
        'Yb': [3.2, 0, 0.2],
        'Lu': [0.8, 0, 0.2],
        'Hf': [3, 0, 0.3],
        'Ta': [2, 0, 0.2],
        'W': [1.25, 1, 0.3],
        'Re': [0.0007, 1, 0.1],
        'Os': [0.0015, 2, 0.1],
        'Ir': [0.001, 1, 0.1],
        'Pt': [0.005, 0, 0.1],
        'Au': [0.004, 0, 0.2],
        'Hg': [0.085, 3, 0.1],
        'Tl': [0.85, 3, 0.1],
        'Pb': [14, 3, 0.3],
        'Bi': [0.009, 1, 0.3],
        'Th': [9.6, 1, 0.1],
        'U': [2.7, 2, 0.1]
    }

    # Convert to DataFrame
    element_data = []
    for element, values in sustainability_data.items():
        # Calculate normalized abundance (log scale to handle large range)
        if values[0] > 0:
            abundance_normalized = min(1.0, max(0.1, 0.1 + 0.3 * np.log10(values[0] + 1) / 6))
        else:
            abundance_normalized = 0.1
        
        # Calculate toxicity impact (inverse of toxicity rating)
        toxicity_impact = 1.0 - (values[1] / 3.0)
        
        # Calculate overall sustainability score
        sustainability_score = (abundance_normalized * 0.4 + 
                               toxicity_impact * 0.3 + 
                               values[2] * 0.3)  # Recyclability
        
        element_data.append({
            'element': element,
            'abundance_ppm': values[0],
            'toxicity_rating': values[1],
            'recyclability_score': values[2],
            'abundance_score': abundance_normalized,
            'toxicity_impact': toxicity_impact,
            'sustainability_score': sustainability_score
        })

    sustainability_df = pd.DataFrame(element_data)
    sustainability_df.to_csv('data/sustainability/element_sustainability_detailed.csv', index=False)
    print(f"Saved detailed sustainability data for {len(sustainability_df)} elements")

if __name__ == "__main__":
    main()