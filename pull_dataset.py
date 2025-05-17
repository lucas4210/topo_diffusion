#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to obtain and process datasets for sustainable topological quantum materials research.
This script downloads and processes data from:
1. Materials Project (structures, formation energies, electronic properties)
2. Element sustainability metrics (abundance, toxicity)
3. Calculated topological properties

"""

import os
import sys
import json
import time
import shutil
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure, Element
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
MP_API_KEY = os.getenv("MP_API_KEY", "")  # Get API key from .env file

# Optimized data volume caps
MAX_MATERIALS = 40000  # Cap at 40,000 materials for comprehensive coverage
MAX_TOPO_MATERIALS = 8000  # Prioritize up to 8,000 potential topological materials
TRAINING_SUBSET_SIZE = 15000  # Create subset for faster iteration during development

# Create base directory
def create_directory_structure():
    """Create the directory structure for storing datasets."""
    directories = [
        BASE_DATA_DIR,
        os.path.join(BASE_DATA_DIR, 'materials_project'),
        os.path.join(BASE_DATA_DIR, 'sustainability'),
        os.path.join(BASE_DATA_DIR, 'topological'),
        os.path.join(BASE_DATA_DIR, 'processed'),
        os.path.join(BASE_DATA_DIR, 'generated')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_materials_project_data():
    """
    Download data from Materials Project including structures,
    formation energies, and electronic properties.
    """
    print("\n--- Downloading Materials Project Data ---")
    mp_data_dir = os.path.join(BASE_DATA_DIR, 'materials_project')
    os.makedirs(mp_data_dir, exist_ok=True)
    
    # Check if existing materials data is available
    existing_csv = os.path.join(mp_data_dir, 'materials_summary.csv')
    existing_json = os.path.join(mp_data_dir, 'all_materials.json')
    
    if os.path.exists(existing_csv) and os.path.getsize(existing_csv) > 0:
        print(f"Found existing materials data at {existing_csv}")
        print("Using existing data instead of downloading new data.")
        return True
    
    # Check if API key is set
    if MP_API_KEY == os.getenv("MP_API_KEY", ""):
        print("Please set your Materials Project API key in the script.")
        print("Register at https://materialsproject.org/dashboard and generate an API key.")
        return False
    
    try:
        # Try to get pymatgen version without relying on __version__
        import pymatgen
        try:
            # Try different methods to get version
            if hasattr(pymatgen, 'version'):
                version = pymatgen.version
            elif hasattr(pymatgen, '__version__'):
                version = pymatgen.__version__
            else:
                try:
                    import pkg_resources
                    version = pkg_resources.get_distribution("pymatgen").version
                except:
                    version = "unknown"
            print(f"Using pymatgen version: {version}")
        except:
            print("Could not determine pymatgen version")
        
        # Initialize the Materials Project API client
        from pymatgen.ext.matproj import MPRester
        mpr = MPRester(MP_API_KEY)
        
        # Define properties to fetch
        properties = [
            "material_id", 
            "formula", 
            "structure", 
            "formation_energy_per_atom",
            "band_gap", 
            "e_above_hull", 
            "spacegroup",
            "elements"
        ]
        
        # Try to get data using different methods for backward compatibility
        print("Attempting to retrieve data from Materials Project API...")
        materials = []
        
        # Start with elements commonly found in topological materials
        topo_elements = ['Bi', 'Sb', 'Te', 'Se', 'Sn', 'Pb', 'Hg', 'Tl']
        print(f"Fetching up to {MAX_TOPO_MATERIALS} potential topological materials...")
        
        # Try different API methods
        success = False
        api_methods = {}
        
        # Check if mpr.query exists and is callable
        if hasattr(mpr, 'query') and callable(getattr(mpr, 'query')):
            api_methods['query'] = True
        else:
            api_methods['query'] = False
            
        # Check if mpr.summary.search exists and is callable
        if hasattr(mpr, 'summary') and hasattr(getattr(mpr, 'summary', None), 'search') and callable(getattr(getattr(mpr, 'summary', None), 'search', None)):
            api_methods['summary.search'] = True
        else:
            api_methods['summary.search'] = False
            
        # Check if mpr.materials.summary.search exists and is callable
        if (hasattr(mpr, 'materials') and 
            hasattr(getattr(mpr, 'materials', None), 'summary') and 
            hasattr(getattr(getattr(mpr, 'materials', None), 'summary', None), 'search') and 
            callable(getattr(getattr(getattr(mpr, 'materials', None), 'summary', None), 'search', None))):
            api_methods['materials.summary.search'] = True
        else:
            api_methods['materials.summary.search'] = False
        
        print("Available API methods:", api_methods)
        
        # Try different methods in order
        topo_criteria = {
            "elements": {"$in": topo_elements},
            "band_gap": {"$gte": 0, "$lt": 0.5},  # Small band gap
            "e_above_hull": {"$lt": 0.1}          # Reasonably stable
        }
        
        # Try primary methods
        if api_methods.get('query', False):
            try:
                print("Using query method...")
                topo_materials = mpr.query(topo_criteria, properties, limit=MAX_TOPO_MATERIALS)
                print(f"Retrieved {len(topo_materials)} materials")
                materials.extend(topo_materials)
                success = True
            except Exception as e:
                print(f"query method failed: {e}")
        
        if not success and api_methods.get('summary.search', False):
            try:
                print("Using summary.search method...")
                topo_materials = mpr.summary.search(criteria=topo_criteria, fields=properties, limit=MAX_TOPO_MATERIALS)
                print(f"Retrieved {len(topo_materials)} materials")
                materials.extend(topo_materials)
                success = True
            except Exception as e:
                print(f"summary.search method failed: {e}")
                
        if not success and api_methods.get('materials.summary.search', False):
            try:
                print("Using materials.summary.search method...")
                topo_materials = mpr.materials.summary.search(criteria=topo_criteria, fields=properties, limit=MAX_TOPO_MATERIALS)
                print(f"Retrieved {len(topo_materials)} materials")
                materials.extend(topo_materials)
                success = True
            except Exception as e:
                print(f"materials.summary.search method failed: {e}")
        
        # If we didn't get any materials, try a simpler query
        if len(materials) == 0:
            print("Trying a simpler query to get at least some materials...")
            simple_criteria = {"nelements": {"$lte": 3}, "e_above_hull": {"$lt": 0.1}}
            
            if api_methods.get('query', False):
                try:
                    simple_materials = mpr.query(simple_criteria, properties, limit=100)
                    print(f"Retrieved {len(simple_materials)} materials with simple query")
                    materials.extend(simple_materials)
                except Exception as e:
                    print(f"Simple query failed: {e}")
            
            if len(materials) == 0 and api_methods.get('summary.search', False):
                try:
                    simple_materials = mpr.summary.search(criteria=simple_criteria, fields=properties, limit=100)
                    print(f"Retrieved {len(simple_materials)} materials with simple query")
                    materials.extend(simple_materials)
                except Exception as e:
                    print(f"Simple summary.search failed: {e}")
            
            if len(materials) == 0 and api_methods.get('materials.summary.search', False):
                try:
                    simple_materials = mpr.materials.summary.search(criteria=simple_criteria, fields=properties, limit=100)
                    print(f"Retrieved {len(simple_materials)} materials with simple query")
                    materials.extend(simple_materials)
                except Exception as e:
                    print(f"Simple materials.summary.search failed: {e}")
        
        # If we still don't have materials, try to get specific IDs that are likely to exist
        if len(materials) == 0:
            print("Trying to retrieve specific common materials by ID...")
            common_ids = ["mp-149", "mp-13", "mp-22862", "mp-568345", "mp-10044"]
            
            for material_id in common_ids:
                try:
                    if hasattr(mpr, 'get_data') and callable(getattr(mpr, 'get_data')):
                        data = mpr.get_data(material_id)
                        if data:
                            materials.extend(data)
                            print(f"Retrieved {material_id}")
                    elif hasattr(mpr, 'get_entry_by_material_id') and callable(getattr(mpr, 'get_entry_by_material_id')):
                        entry = mpr.get_entry_by_material_id(material_id)
                        if entry:
                            materials.append({
                                'material_id': material_id,
                                'formula': entry.composition.formula,
                                'structure': entry.structure,
                                'formation_energy_per_atom': entry.energy_per_atom,
                                'band_gap': getattr(entry, 'band_gap', 0.0),
                                'e_above_hull': getattr(entry, 'e_above_hull', 0.0),
                                'elements': list(entry.composition.get_el_amt_dict().keys())
                            })
                            print(f"Retrieved {material_id} using entry")
                except Exception as e:
                    print(f"Failed to retrieve {material_id}: {e}")
        
        # If we got materials, process them
        if len(materials) > 0:
            print(f"Successfully retrieved {len(materials)} materials. Processing...")
            
            # Create structures directory if it doesn't exist
            poscar_dir = os.path.join(mp_data_dir, 'structures')
            os.makedirs(poscar_dir, exist_ok=True)
            
            # Process and save materials
            processed_materials = []
            for material in tqdm(materials, desc="Processing materials"):
                try:
                    # Deep copy of material to avoid modifying original
                    material_copy = {}
                    for key, value in material.items():
                        if key == 'structure':
                            # Handle structure conversion
                            if hasattr(value, 'as_dict'):
                                material_copy[key] = value.as_dict()
                            elif isinstance(value, dict):
                                material_copy[key] = value
                            else:
                                print(f"Unknown structure type: {type(value)}")
                                continue
                        else:
                            material_copy[key] = value
                    
                    processed_materials.append(material_copy)
                    
                    # Save structure as POSCAR file
                    material_id = material.get('material_id', f"unknown-{len(processed_materials)}")
                    poscar_file = os.path.join(poscar_dir, f"{material_id}.vasp")
                    
                    # Convert to Structure if needed
                    if hasattr(material.get('structure', None), 'as_dict'):
                        structure = material['structure']
                    else:
                        try:
                            from pymatgen.core import Structure
                            structure = Structure.from_dict(material['structure'])
                        except Exception as e:
                            print(f"Error converting structure for {material_id}: {e}")
                            continue
                    
                    # Write POSCAR file
                    from pymatgen.io.vasp import Poscar
                    Poscar(structure).write_file(poscar_file)
                    
                except Exception as e:
                    print(f"Error processing material: {e}")
                    continue
            
            # Save all materials to JSON
            all_materials_file = os.path.join(mp_data_dir, 'all_materials.json')
            with open(all_materials_file, 'w') as f:
                json.dump(processed_materials, f)
            
            # Create CSV with basic properties
            materials_data = []
            for m in processed_materials:
                try:
                    material_dict = {
                        'material_id': m.get('material_id', 'unknown'),
                        'formula': m.get('formula', 'unknown'),
                        'band_gap': float(m.get('band_gap', 0.0)),
                        'formation_energy': float(m.get('formation_energy_per_atom', 0.0)),
                        'e_above_hull': float(m.get('e_above_hull', 0.0)),
                    }
                    
                    # Handle spacegroup
                    if isinstance(m.get('spacegroup', None), dict):
                        material_dict['spacegroup'] = m['spacegroup'].get('symbol', 'unknown')
                    else:
                        material_dict['spacegroup'] = str(m.get('spacegroup', 'unknown'))
                    
                    # Handle elements
                    if isinstance(m.get('elements', None), list):
                        material_dict['elements'] = ','.join(m['elements'])
                    else:
                        material_dict['elements'] = ','.join(str(m.get('elements', '')).split())
                    
                    materials_data.append(material_dict)
                except Exception as e:
                    print(f"Error creating CSV entry: {e}")
                    continue
            
            materials_df = pd.DataFrame(materials_data)
            csv_path = os.path.join(mp_data_dir, 'materials_summary.csv')
            materials_df.to_csv(csv_path, index=False)
            print(f"Saved materials summary to {csv_path}")
            
            # Create training subset
            training_size = min(TRAINING_SUBSET_SIZE, len(materials_df))
            training_subset = materials_df.sample(n=training_size, random_state=42)
            subset_path = os.path.join(mp_data_dir, 'training_subset.csv')
            training_subset.to_csv(subset_path, index=False)
            print(f"Saved training subset to {subset_path}")
            
            # Copy structure files for training subset
            subset_dir = os.path.join(mp_data_dir, 'training_subset_structures')
            os.makedirs(subset_dir, exist_ok=True)
            
            for material_id in tqdm(training_subset['material_id'], desc="Copying training subset structures"):
                src_file = os.path.join(poscar_dir, f"{material_id}.vasp")
                dst_file = os.path.join(subset_dir, f"{material_id}.vasp")
                if os.path.exists(src_file):
                    shutil.copy(src_file, dst_file)
            
            return True
        else:
            print("Could not retrieve any materials from the API.")
            print("Please check your API key or internet connection.")
            
            # Create empty CSV file to allow pipeline to continue
            materials_df = pd.DataFrame(columns=['material_id', 'formula', 'band_gap', 
                                              'formation_energy', 'e_above_hull',
                                              'spacegroup', 'elements'])
            csv_path = os.path.join(mp_data_dir, 'materials_summary.csv')
            materials_df.to_csv(csv_path, index=False)
            print(f"Created empty materials file at {csv_path}")
            
            return False
                
    except Exception as e:
        print(f"Error downloading Materials Project data: {e}")
        import traceback
        traceback.print_exc()
        
        # Create empty CSV file to allow pipeline to continue
        materials_df = pd.DataFrame(columns=['material_id', 'formula', 'band_gap', 
                                          'formation_energy', 'e_above_hull',
                                          'spacegroup', 'elements'])
        csv_path = os.path.join(mp_data_dir, 'materials_summary.csv')
        materials_df.to_csv(csv_path, index=False)
        print(f"Created empty materials file at {csv_path}")
        
        return False

def create_element_sustainability_dataset():
    """
    Create a dataset with element sustainability metrics.
    Combines crustal abundance and basic toxicity information.
    """
    print("\n--- Creating Element Sustainability Dataset ---")
    sustainability_dir = os.path.join(BASE_DATA_DIR, 'sustainability')
    
    # Create a basic dataset with crustal abundance
    # Data sourced from various references including USGS and periodictable.com
    
    # First, try to download abundance data from periodictable.com
    try:
        print("Downloading crustal abundance data...")
        abundance_url = "https://periodictable.com/Properties/A/CrustAbundance.html"
        response = requests.get(abundance_url)
        
        if response.status_code == 200:
            # Very basic scraping
            abundance_data = {}
            
            for element in Element:
                symbol = element.symbol
                # Look for the pattern in the HTML
                pattern = f'<tr.*?<td.*?>{symbol}</td>.*?<td.*?>([0-9.]+)</td>'
                import re
                match = re.search(pattern, response.text)
                
                if match:
                    abundance_data[symbol] = float(match.group(1))
                else:
                    # Default value if not found
                    abundance_data[symbol] = 0.001
            
            print(f"Found abundance data for {len(abundance_data)} elements")
        else:
            # Fallback to a basic dataset if download fails
            print("Failed to download abundance data, using fallback dataset")
            abundance_data = None
    except Exception as e:
        print(f"Error downloading abundance data: {e}")
        abundance_data = None
    
    # If download failed, use a basic predefined dataset
    if abundance_data is None:
        # Create a basic dataset with estimated values for common elements
        # Values in parts per million (ppm) in Earth's crust
        abundance_data = {
            'H': 1400, 'He': 0.008, 'Li': 20, 'Be': 2.8, 'B': 10, 'C': 200,
            'N': 20, 'O': 461000, 'F': 585, 'Ne': 0.005, 'Na': 23600, 'Mg': 23300,
            'Al': 82300, 'Si': 282000, 'P': 1050, 'S': 350, 'Cl': 145, 'Ar': 3.5,
            'K': 20900, 'Ca': 41500, 'Sc': 22, 'Ti': 5650, 'V': 120, 'Cr': 102,
            'Mn': 950, 'Fe': 56300, 'Co': 25, 'Ni': 84, 'Cu': 60, 'Zn': 70,
            'Ga': 19, 'Ge': 1.5, 'As': 1.8, 'Se': 0.05, 'Br': 2.4, 'Kr': 0.001,
            'Rb': 90, 'Sr': 370, 'Y': 33, 'Zr': 165, 'Nb': 20, 'Mo': 1.2,
            'Tc': 0, 'Ru': 0.001, 'Rh': 0.001, 'Pd': 0.015, 'Ag': 0.075, 'Cd': 0.15,
            'In': 0.25, 'Sn': 2.3, 'Sb': 0.2, 'Te': 0.001, 'I': 0.45, 'Xe': 0.0003,
            'Cs': 3, 'Ba': 425, 'La': 39, 'Ce': 66.5, 'Pr': 9.2, 'Nd': 41.5,
            'Pm': 0, 'Sm': 7.05, 'Eu': 2, 'Gd': 6.2, 'Tb': 1.2, 'Dy': 5.2,
            'Ho': 1.3, 'Er': 3.5, 'Tm': 0.52, 'Yb': 3.2, 'Lu': 0.8, 'Hf': 3,
            'Ta': 2, 'W': 1.25, 'Re': 0.0007, 'Os': 0.0015, 'Ir': 0.001, 'Pt': 0.005,
            'Au': 0.004, 'Hg': 0.085, 'Tl': 0.85, 'Pb': 14, 'Bi': 0.009, 'Po': 0,
            'At': 0, 'Rn': 0, 'Fr': 0, 'Ra': 0, 'Ac': 0, 'Th': 9.6, 'Pa': 0,
            'U': 2.7, 'Np': 0, 'Pu': 0
        }

    # Basic toxicity ratings (0-3 scale, where 0 is non-toxic, 3 is highly toxic)
    # This is a simplified approximation - real environmental assessment would use more detailed data
    toxicity_ratings = {
        'H': 0, 'He': 0, 'Li': 1, 'Be': 3, 'B': 1, 'C': 0, 'N': 0, 'O': 0,
        'F': 2, 'Ne': 0, 'Na': 1, 'Mg': 0, 'Al': 1, 'Si': 0, 'P': 2, 'S': 1,
        'Cl': 2, 'Ar': 0, 'K': 1, 'Ca': 0, 'Sc': 1, 'Ti': 1, 'V': 2, 'Cr': 2,
        'Mn': 1, 'Fe': 0, 'Co': 2, 'Ni': 2, 'Cu': 1, 'Zn': 1, 'Ga': 1, 'Ge': 1,
        'As': 3, 'Se': 2, 'Br': 2, 'Kr': 0, 'Rb': 1, 'Sr': 1, 'Y': 1, 'Zr': 1,
        'Nb': 1, 'Mo': 1, 'Tc': 2, 'Ru': 2, 'Rh': 1, 'Pd': 1, 'Ag': 1, 'Cd': 3,
        'In': 1, 'Sn': 1, 'Sb': 2, 'Te': 2, 'I': 1, 'Xe': 0, 'Cs': 1, 'Ba': 1,
        'La': 1, 'Ce': 1, 'Pr': 1, 'Nd': 1, 'Pm': 2, 'Sm': 1, 'Eu': 1, 'Gd': 1,
        'Tb': 1, 'Dy': 1, 'Ho': 1, 'Er': 1, 'Tm': 1, 'Yb': 1, 'Lu': 1, 'Hf': 1,
        'Ta': 1, 'W': 1, 'Re': 1, 'Os': 2, 'Ir': 1, 'Pt': 1, 'Au': 0, 'Hg': 3,
        'Tl': 3, 'Pb': 3, 'Bi': 1, 'Po': 3, 'At': 3, 'Rn': 3, 'Fr': 2, 'Ra': 3,
        'Ac': 3, 'Th': 3, 'Pa': 3, 'U': 3, 'Np': 3, 'Pu': 3
    }
    
    # Create a pandas DataFrame
    elements = sorted(abundance_data.keys())
    element_data = []
    
    for element in elements:
        element_data.append({
            'element': element,
            'abundance_ppm': abundance_data.get(element, 0.001),
            'toxicity_rating': toxicity_ratings.get(element, 1)
        })
    
    element_df = pd.DataFrame(element_data)
    
    # Calculate normalized abundance score (higher is better)
    element_df['abundance_score'] = element_df['abundance_ppm'].apply(lambda x: np.log10(x + 0.001) if x > 0 else -3)
    element_df['abundance_score'] = element_df['abundance_score'].rank(pct=True)
    
    # Calculate overall sustainability score (higher is better)
    element_df['sustainability_score'] = element_df['abundance_score'] * (3 - element_df['toxicity_rating']) / 3
    
    # Save to CSV
    output_file = os.path.join(sustainability_dir, 'element_sustainability.csv')
    element_df.to_csv(output_file, index=False)
    print(f"Saved element sustainability data to {output_file}")
    
    return element_df

def extract_topological_features():
    """
    Extract or calculate basic topological properties for materials.
    For a full implementation, this would involve DFT calculations or machine learning predictions.
    Here we implement a simplified heuristic approach based on known properties.
    """
    print("\n--- Extracting Topological Features ---")
    mp_data_dir = os.path.join(BASE_DATA_DIR, 'materials_project')
    topo_dir = os.path.join(BASE_DATA_DIR, 'topological')
    os.makedirs(topo_dir, exist_ok=True)
    
    # Load materials data
    try:
        materials_df = pd.read_csv(os.path.join(mp_data_dir, 'materials_summary.csv'))
        print(f"Loaded {len(materials_df)} materials for topological analysis")
    except Exception as e:
        print(f"Error loading materials data: {e}")
        print("Please run download_materials_project_data() first")
        
        # Create a minimal dataset to allow the script to continue
        print("Creating minimal topological dataset to allow script to continue...")
        columns = ['material_id', 'formula', 'band_gap', 'formation_energy', 
                  'e_above_hull', 'spacegroup', 'elements', 
                  'contains_topo_elements', 'is_potential_ti', 'is_potential_dsm', 'topo_score']
        
        materials_df = pd.DataFrame(columns=columns)
        # Add at least one row with dummy data
        materials_df.loc[0] = ['dummy-0', 'Si2O', 0.1, -2.0, 0.0, 'P1', 'Si,O', 
                              False, False, False, 0.0]
        
        # Save this minimal dataset
        materials_df.to_csv(os.path.join(topo_dir, 'topological_predictions.csv'), index=False)
        return materials_df
    
    # Simple heuristics for potential topological materials
    # In a real implementation, this would be replaced with ML predictions or DFT calculations
    print("Applying heuristics to identify potential topological materials...")
    
    # Criteria that might indicate topological properties:
    # 1. Materials with band gaps in the range 0-0.3 eV (potential topological insulators)
    # 2. Materials with specific elements known to be common in topological materials
    # 3. Materials with certain space groups that often host topological phases
    
    # Elements often found in topological materials
    topo_elements = ['Bi', 'Sb', 'Te', 'Se', 'Sn', 'Pb', 'Hg', 'Tl']
    
    # Space groups that often host topological phases
    # This is a simplified approximation
    interesting_sg_numbers = [2, 164, 166, 176, 187, 189, 191, 194, 221, 224, 229]
    
    # Check if required columns exist, and if not, add them with default values
    if 'elements' not in materials_df.columns:
        materials_df['elements'] = 'unknown'
    
    # Add topological features
    materials_df['contains_topo_elements'] = materials_df['elements'].apply(
        lambda x: any(elem in str(x).split(',') for elem in topo_elements)
    )
    
    # Make sure band_gap column is numeric
    materials_df['band_gap'] = pd.to_numeric(materials_df['band_gap'], errors='coerce').fillna(0)
    
    materials_df['is_potential_ti'] = (
        (materials_df['band_gap'] > 0) & 
        (materials_df['band_gap'] < 0.3) & 
        materials_df['contains_topo_elements']
    )
    
    materials_df['is_potential_dsm'] = (
        (materials_df['band_gap'] < 0.01) & 
        materials_df['contains_topo_elements']
    )
    
    # Calculate a simple "topological likelihood score"
    # This is just a heuristic for demonstration
    materials_df['topo_score'] = 0.0
    
    # Increase score for small band gap
    materials_df.loc[materials_df['band_gap'] < 0.3, 'topo_score'] += 0.3
    materials_df.loc[materials_df['band_gap'] < 0.1, 'topo_score'] += 0.2
    
    # Increase score for containing topological elements
    materials_df.loc[materials_df['contains_topo_elements'], 'topo_score'] += 0.3
    
    # Adjust to ensure scores are between 0 and 1
    materials_df['topo_score'] = materials_df['topo_score'].clip(0, 1)
    
    # Save results
    output_file = os.path.join(topo_dir, 'topological_predictions.csv')
    materials_df.to_csv(output_file, index=False)
    print(f"Saved topological predictions to {output_file}")
    
    # Create a filtered dataset of the most promising topological materials
    # Use a safer approach to filter data
    stability_filter = materials_df['e_above_hull'] < 0.05 if 'e_above_hull' in materials_df.columns else True
    
    top_candidates = materials_df[
        (materials_df['topo_score'] > 0.5) & stability_filter
    ].sort_values('topo_score', ascending=False)
    
    # If we have more candidates than our cap, limit to the most promising ones
    if len(top_candidates) > MAX_TOPO_MATERIALS:
        print(f"Limiting top candidates to {MAX_TOPO_MATERIALS} materials")
        top_candidates = top_candidates.head(MAX_TOPO_MATERIALS)
    
    top_output_file = os.path.join(topo_dir, 'top_topological_candidates.csv')
    top_candidates.to_csv(top_output_file, index=False)
    print(f"Saved top topological candidates to {top_output_file}")
    print(f"Identified {len(top_candidates)} promising topological material candidates")
    
    # Also create a subset of structures specifically for topological materials
    topo_struct_dir = os.path.join(topo_dir, 'structures')
    os.makedirs(topo_struct_dir, exist_ok=True)
    
    for material_id in tqdm(top_candidates['material_id'], desc="Copying topological material structures"):
        src_file = os.path.join(mp_data_dir, 'structures', f"{material_id}.vasp")
        dst_file = os.path.join(topo_struct_dir, f"{material_id}.vasp")
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
    
    return materials_df

def create_combined_dataset():
    """Combine all datasets into a single dataset for model training."""
    print("\n--- Creating Combined Dataset ---")
    processed_dir = os.path.join(BASE_DATA_DIR, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load individual datasets
    try:
        # Materials data (try to load topological predictions which should have all base data plus topo features)
        topo_data_file = os.path.join(BASE_DATA_DIR, 'topological', 'topological_predictions.csv')
        if os.path.exists(topo_data_file):
            combined_df = pd.read_csv(topo_data_file)
            print(f"Loaded topological predictions for {len(combined_df)} materials")
        else:
            # Fall back to base materials data
            mp_data_file = os.path.join(BASE_DATA_DIR, 'materials_project', 'materials_summary.csv')
            combined_df = pd.read_csv(mp_data_file)
            print(f"Loaded {len(combined_df)} materials")
            
            # Check if topo_score exists, if not add it with zeros
            if 'topo_score' not in combined_df.columns:
                combined_df['topo_score'] = 0.0
                print("Warning: No topological scores found. Added default values.")
        
        # Sustainability data
        sust_data_file = os.path.join(BASE_DATA_DIR, 'sustainability', 'element_sustainability.csv')
        if os.path.exists(sust_data_file):
            sust_df = pd.read_csv(sust_data_file)
            print(f"Loaded sustainability data for {len(sust_df)} elements")
        else:
            # Create minimal sustainability data
            print("Warning: Sustainability data not found. Creating minimal dataset.")
            sust_df = pd.DataFrame({
                'element': ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O'],
                'abundance_ppm': [1400, 0.008, 20, 2.8, 10, 200, 20, 461000],
                'toxicity_rating': [0, 0, 1, 3, 1, 0, 0, 0],
                'abundance_score': [0.5, 0.1, 0.4, 0.2, 0.3, 0.6, 0.4, 0.9],
                'sustainability_score': [0.5, 0.1, 0.3, 0.05, 0.2, 0.6, 0.4, 0.9]
            })
    except Exception as e:
        print(f"Error loading required datasets: {e}")
        print("Please run all data collection functions first")
        
        # Create minimal datasets to allow the script to continue
        print("Creating minimal combined dataset to allow script to continue...")
        
        # Create a minimal combined dataset
        combined_df = pd.DataFrame({
            'material_id': ['dummy-1', 'dummy-2'],
            'formula': ['Si2O', 'Fe2O3'],
            'band_gap': [0.1, 1.5],
            'formation_energy': [-2.0, -3.0],
            'e_above_hull': [0.0, 0.01],
            'spacegroup': ['P1', 'R-3c'],
            'elements': ['Si,O', 'Fe,O'],
            'topo_score': [0.5, 0.1]
        })
        
        # Create minimal sustainability data
        sust_df = pd.DataFrame({
            'element': ['Si', 'O', 'Fe'],
            'sustainability_score': [0.6, 0.9, 0.4]
        })
        
        # Save the minimal dataset
        combined_df.to_csv(os.path.join(processed_dir, 'combined_dataset.csv'), index=False)
        return combined_df
    
    # Add sustainability metrics by calculating weighted average of element sustainability
    print("Calculating composition-based sustainability metrics...")
    
    # Create a lookup dictionary for element sustainability
    element_sus_dict = dict(zip(sust_df['element'], sust_df.get('sustainability_score', sust_df.get('abundance_score', [0.5] * len(sust_df)))))
    
    # Function to calculate sustainability of a composition
    def calc_sustainability(element_string):
        try:
            elements = str(element_string).split(',')
            # Simple average of sustainability scores
            # In a real implementation, this would use the actual composition fractions
            scores = [element_sus_dict.get(element.strip(), 0.2) for element in elements]
            return sum(scores) / len(scores) if scores else 0.0
        except Exception as e:
            print(f"Error calculating sustainability: {e}")
            return 0.2  # Default value
    
    combined_df['sustainability_score'] = combined_df['elements'].apply(calc_sustainability)
    
    # Create a combined score that balances stability, topological interest, and sustainability
    print("Calculating combined scores...")
    
    # Make sure required columns exist with appropriate defaults
    if 'e_above_hull' not in combined_df.columns:
        combined_df['e_above_hull'] = 0.0
    
    # Normalize e_above_hull (lower is better, so invert)
    max_e_above_hull = combined_df['e_above_hull'].max()
    if max_e_above_hull > 0:
        combined_df['stability_score'] = 1.0 - (combined_df['e_above_hull'] / max_e_above_hull)
    else:
        combined_df['stability_score'] = 1.0
    
    # Combined score with adjustable weights
    weights = {
        'stability': 0.3,
        'topological': 0.4,
        'sustainability': 0.3
    }
    
    combined_df['combined_score'] = (
        weights['stability'] * combined_df['stability_score'] +
        weights['topological'] * combined_df['topo_score'] +
        weights['sustainability'] * combined_df['sustainability_score']
    )
    
    # Normalize to 0-1 range
    max_score = combined_df['combined_score'].max()
    if max_score > 0:
        combined_df['combined_score'] = combined_df['combined_score'] / max_score
    
    # Save the combined dataset
    output_file = os.path.join(processed_dir, 'combined_dataset.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined dataset to {output_file}")
    
    # Create a filtered dataset of the most promising sustainable topological materials
    top_candidates = combined_df[
        combined_df['combined_score'] > 0.7
    ].sort_values('combined_score', ascending=False)
    
    top_output_file = os.path.join(processed_dir, 'top_sustainable_topological_candidates.csv')
    top_candidates.to_csv(top_output_file, index=False)
    print(f"Saved top sustainable topological candidates to {top_output_file}")
    print(f"Identified {len(top_candidates)} promising sustainable topological material candidates")
    
    # Create HDF5 storage for more efficient data access if h5py is available
    try:
        import h5py
        print("Creating HDF5 dataset for efficient training...")
        
        h5_file = os.path.join(processed_dir, 'training_data.h5')
        with h5py.File(h5_file, 'w') as f:
            # Store property data
            properties = f.create_group('properties')
            
            # Get columns that are numerical for storage
            numeric_columns = combined_df.select_dtypes(include=['number']).columns.tolist()
            
            # Store material ids as a reference
            dt = h5py.special_dtype(vlen=str)
            try:
                material_ids = np.array(combined_df['material_id'].tolist(), dtype=object)
                properties.create_dataset('material_id', data=material_ids, dtype=dt)
            except Exception as e:
                print(f"Error storing material_ids: {e}")
            
            # Store formulas similarly
            try:
                formulas = np.array(combined_df['formula'].tolist(), dtype=object)
                properties.create_dataset('formula', data=formulas, dtype=dt)
            except Exception as e:
                print(f"Error storing formulas: {e}")
            
            # Store numeric properties
            for prop in numeric_columns:
                try:
                    properties.create_dataset(prop, data=combined_df[prop].values)
                except Exception as e:
                    print(f"Error storing {prop}: {e}")
            
            # Create training subset indices if available
            subset_file = os.path.join(BASE_DATA_DIR, 'materials_project', 'training_subset.csv')
            if os.path.exists(subset_file):
                training_subset = pd.read_csv(subset_file)
                training_ids = set(training_subset['material_id'])
                
                is_in_training = np.array([mid in training_ids for mid in combined_df['material_id']])
                properties.create_dataset('is_in_training', data=is_in_training)
            
            # Create topological candidates indices if available
            topo_file = os.path.join(BASE_DATA_DIR, 'topological', 'top_topological_candidates.csv')
            if os.path.exists(topo_file):
                topo_candidates = pd.read_csv(topo_file)
                topo_ids = set(topo_candidates['material_id'])
                
                is_topo_candidate = np.array([mid in topo_ids for mid in combined_df['material_id']])
                properties.create_dataset('is_topo_candidate', data=is_topo_candidate)
            
        print(f"Created HDF5 dataset at {h5_file}")
    except ImportError:
        print("h5py not installed. Skipping HDF5 dataset creation.")
        print("For better performance, consider installing h5py: pip install h5py")
    except Exception as e:
        print(f"Error creating HDF5 dataset: {e}")
    
    return combined_df

def main():
    """Main function to run the data collection and processing pipeline."""
    print("Starting data collection and processing for sustainable topological materials research")
    print(f"Data volume settings:")
    print(f"- Maximum materials: {MAX_MATERIALS}")
    print(f"- Maximum topological materials: {MAX_TOPO_MATERIALS}")
    print(f"- Training subset size: {TRAINING_SUBSET_SIZE}")
    
    # Create directory structure
    create_directory_structure()
    
    # Download and process datasets
    download_materials_project_data()
    create_element_sustainability_dataset()
    extract_topological_features()
    create_combined_dataset()
    
    # Perform data quality and completeness check
    check_data_completeness()
    
    print("\nData collection and processing complete!")
    print(f"All data saved to {BASE_DATA_DIR}")
    print("\nNext steps:")
    print("1. Review the collected datasets")
    print("2. Start with the training subset for model development")
    print("3. Scale up to the full dataset for final model training")

def check_data_completeness():
    """Check that all expected data files exist and have appropriate content."""
    print("\n--- Checking Data Completeness ---")
    
    # Define expected files
    expected_files = [
        os.path.join(BASE_DATA_DIR, 'materials_project', 'materials_summary.csv'),
        os.path.join(BASE_DATA_DIR, 'materials_project', 'training_subset.csv'),
        os.path.join(BASE_DATA_DIR, 'sustainability', 'element_sustainability.csv'),
        os.path.join(BASE_DATA_DIR, 'topological', 'topological_predictions.csv'),
        os.path.join(BASE_DATA_DIR, 'topological', 'top_topological_candidates.csv'),
        os.path.join(BASE_DATA_DIR, 'processed', 'combined_dataset.csv'),
        os.path.join(BASE_DATA_DIR, 'processed', 'top_sustainable_topological_candidates.csv'),
    ]
    
    missing_files = []
    empty_files = []
    
    # Check each file
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        elif os.path.getsize(file_path) < 100:  # Arbitrary small size to detect essentially empty files
            empty_files.append(file_path)
    
    # Check structure directories
    structure_dirs = [
        os.path.join(BASE_DATA_DIR, 'materials_project', 'structures'),
        os.path.join(BASE_DATA_DIR, 'materials_project', 'training_subset_structures'),
        os.path.join(BASE_DATA_DIR, 'topological', 'structures')
    ]
    
    empty_dirs = []
    for dir_path in structure_dirs:
        if not os.path.exists(dir_path):
            missing_files.append(dir_path)
        else:
            files = os.listdir(dir_path)
            if len(files) == 0:
                empty_dirs.append(dir_path)
    
    # Report on data completeness
    if missing_files:
        print("WARNING: The following expected files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    
    if empty_files:
        print("WARNING: The following files exist but appear to be empty:")
        for file_path in empty_files:
            print(f"  - {file_path}")
            
    if empty_dirs:
        print("WARNING: The following directories exist but contain no files:")
        for dir_path in empty_dirs:
            print(f"  - {dir_path}")
    
    if not missing_files and not empty_files and not empty_dirs:
        print("SUCCESS: All expected data files and directories are present and contain data")
    
    # Check data volume
    try:
        # Check materials count
        materials_file = os.path.join(BASE_DATA_DIR, 'materials_project', 'materials_summary.csv')
        if os.path.exists(materials_file):
            materials_df = pd.read_csv(materials_file)
            print(f"Total materials collected: {len(materials_df)} / {MAX_MATERIALS} target")
        
        # Check topological materials count
        topo_file = os.path.join(BASE_DATA_DIR, 'topological', 'top_topological_candidates.csv')
        if os.path.exists(topo_file):
            topo_df = pd.read_csv(topo_file)
            print(f"Topological materials identified: {len(topo_df)} / {MAX_TOPO_MATERIALS} target")
        
        # Check training subset
        subset_file = os.path.join(BASE_DATA_DIR, 'materials_project', 'training_subset.csv')
        if os.path.exists(subset_file):
            subset_df = pd.read_csv(subset_file)
            print(f"Training subset size: {len(subset_df)} / {TRAINING_SUBSET_SIZE} target")
    
    except Exception as e:
        print(f"Error checking data volumes: {e}")
    
    # Calculate total storage used
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(BASE_DATA_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    # Convert to MB or GB for readability
    if total_size > 1024 * 1024 * 1024:
        print(f"Total storage used: {total_size / (1024 * 1024 * 1024):.2f} GB")
    else:
        print(f"Total storage used: {total_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()