# Sustainable Topological Quantum Materials Dataset

This document provides an overview of the dataset structure, contents, and usage guidelines for the Sustainable Topological Quantum Materials project.

## Overview

This dataset is designed for training generative machine learning models to discover novel materials that combine topological properties with sustainability. It contains crystal structures, electronic properties, topological classifications, and sustainability metrics for up to 40,000 materials.

## Dataset Structure

The dataset is organized in a hierarchical directory structure:

```
data/
├── materials_project/           # Raw materials data
│   ├── all_materials.json       # Complete materials dataset in JSON format
│   ├── materials_summary.csv    # Summary of material properties in CSV format
│   ├── training_subset.csv      # Subset of materials for faster model development
│   ├── structures/              # Directory containing crystal structures as VASP POSCAR files
│   └── training_subset_structures/ # Structure files for the training subset
│
├── sustainability/              # Sustainability metrics
│   └── element_sustainability.csv # Element-level sustainability scores
│
├── topological/                 # Topological property data
│   ├── topological_predictions.csv # Predicted topological properties
│   ├── top_topological_candidates.csv # High-scoring topological materials
│   └── structures/              # Structure files for topological materials
│
└── processed/                   # Combined and processed datasets
    ├── combined_dataset.csv     # Complete dataset with all properties
    ├── top_sustainable_topological_candidates.csv # Best candidates
    └── training_data.h5         # HDF5 format for efficient training (if h5py is installed)
```

## Dataset Contents

### Materials Project Data

File: `materials_project/materials_summary.csv`

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| material_id | Materials Project identifier | string | "mp-149" |
| formula | Chemical formula | string | "LiFePO4" |
| band_gap | Electronic band gap in eV | float | 0.23 |
| formation_energy | Formation energy per atom in eV | float | -2.14 |
| e_above_hull | Energy above the convex hull in eV | float | 0.0 |
| spacegroup | Space group symbol | string | "Pnma" |
| elements | Comma-separated list of elements | string | "Li,Fe,P,O" |

### Sustainability Metrics

File: `sustainability/element_sustainability.csv`

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| element | Chemical element symbol | string | "Li" |
| abundance_ppm | Crustal abundance in parts per million | float | 20.0 |
| toxicity_rating | Toxicity score (0-3 scale, higher means more toxic) | int | 1 |
| abundance_score | Normalized abundance score (0-1 scale) | float | 0.72 |
| sustainability_score | Overall sustainability score (0-1 scale) | float | 0.48 |

### Topological Properties

File: `topological/topological_predictions.csv`

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| material_id | Materials Project identifier | string | "mp-149" |
| ... | *[All columns from materials_summary.csv]* | ... | ... |
| contains_topo_elements | Contains elements common in topological materials | bool | true |
| is_potential_ti | Potentially a topological insulator | bool | true |
| is_potential_dsm | Potentially a Dirac/Weyl semimetal | bool | false |
| topo_score | Predicted topological interest score (0-1) | float | 0.75 |

### Combined Dataset

File: `processed/combined_dataset.csv`

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| material_id | Materials Project identifier | string | "mp-149" |
| ... | *[All columns from materials_summary.csv]* | ... | ... |
| ... | *[All columns from topological_predictions.csv]* | ... | ... |
| sustainability_score | Composition-weighted sustainability score | float | 0.65 |
| stability_score | Normalized stability score (inverse of e_above_hull) | float | 0.98 |
| combined_score | Weighted combination of topological, sustainability, and stability scores | float | 0.81 |

### HDF5 Dataset (Optional)

File: `processed/training_data.h5`

Structure:
```
properties/
├── material_id           # Array of material IDs
├── band_gap              # Array of band gaps
├── formation_energy      # Array of formation energies
├── e_above_hull          # Array of hull energies
├── topo_score            # Array of topological scores
├── sustainability_score  # Array of sustainability scores
├── combined_score        # Array of combined scores
├── formula               # Array of chemical formulas
├── is_in_training        # Boolean array indicating training subset
└── is_topo_candidate     # Boolean array indicating topological candidates
```

## Data Volume

The dataset is configured with the following volume caps:

- Total materials: Up to 40,000
- Topological materials: Up to 8,000
- Training subset: 15,000 materials

These caps balance dataset comprehensiveness with computational efficiency.

## Using the Dataset

### For Model Development

1. **Initial Exploration:** Start with the training subset for rapid prototyping:
   ```python
   import pandas as pd
   
   # Load the training subset
   training_df = pd.read_csv('data/materials_project/training_subset.csv')
   
   # Load corresponding structures
   from pymatgen.core import Structure
   
   structures = {}
   for material_id in training_df['material_id']:
       poscar_path = f'data/materials_project/training_subset_structures/{material_id}.vasp'
       structures[material_id] = Structure.from_file(poscar_path)
   ```

2. **Full Dataset Training:** Scale to the complete dataset for final model training:
   ```python
   # Option 1: Using CSV files
   full_df = pd.read_csv('data/processed/combined_dataset.csv')
   
   # Option 2: Using HDF5 for better performance
   import h5py
   
   with h5py.File('data/processed/training_data.h5', 'r') as f:
       # Access properties
       material_ids = f['properties/material_id'][:]
       topo_scores = f['properties/topo_score'][:]
       sustainability_scores = f['properties/sustainability_score'][:]
   ```

3. **Evaluating Generated Materials:** Compare with top candidates:
   ```python
   top_candidates = pd.read_csv('data/processed/top_sustainable_topological_candidates.csv')
   
   # Use as reference for your generated materials
   # You can compare properties, stability, etc.
   ```

### For Graph Neural Networks

If using graph-based models:

```python
from pymatgen.core import Structure
import torch
from torch_geometric.data import Data

def structure_to_graph(structure):
    """Convert pymatgen structure to graph representation."""
    # Extract positions and atomic numbers
    pos = torch.tensor([site.coords for site in structure.sites], dtype=torch.float)
    atomic_numbers = torch.tensor([site.specie.Z for site in structure.sites], dtype=torch.long)
    
    # Create edges based on nearest neighbors (simplistic approach)
    edge_index = []
    for i, site in enumerate(structure.sites):
        neighbors = structure.get_neighbors(site, r=6.0)  # 6 Angstrom radius
        for neighbor in neighbors:
            j = structure.sites.index(neighbor.site)
            edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create graph data object
    graph = Data(x=atomic_numbers, edge_index=edge_index, pos=pos)
    return graph

# Create a dataset of graphs
graphs = []
labels = []

for idx, row in training_df.iterrows():
    material_id = row['material_id']
    structure_path = f'data/materials_project/training_subset_structures/{material_id}.vasp'
    try:
        structure = Structure.from_file(structure_path)
        graph = structure_to_graph(structure)
        graph.y = torch.tensor([row['topo_score'], row['sustainability_score']], dtype=torch.float)
        graphs.append(graph)
        labels.append(row['combined_score'])
    except Exception as e:
        print(f"Error processing {material_id}: {e}")
```

## Important Notes

1. **Structure Files:** Crystal structures are stored in VASP POSCAR format, compatible with `pymatgen` and other materials science libraries.

2. **Topological Prediction:** The topological scores are based on heuristics rather than rigorous calculations. For a production-grade model, replace these with DFT-calculated invariants or trained machine learning predictions.

3. **Sustainability Metrics:** The sustainability scores are based on crustal abundance and basic toxicity. For more rigorous research, consider incorporating more detailed environmental impact metrics.

4. **Memory Requirements:** Working with the full dataset requires approximately:
   - 10-15 GB of storage space
   - 32+ GB of RAM for full processing
   - 16+ GB of GPU memory for training graph neural networks

## Citation

If you use this dataset in your research, please cite:

```
@article{your_name_here2025,
  title={De Novo Design of Sustainable Topological Quantum Materials Using Generative Machine Learning},
  author={Your Name Here and Collaborators},
  journal={Nature Communications},
  year={2025},
  volume={},
  pages={}
}
```

## License

This dataset and accompanying code are made available under the MIT License:

```
MIT License

Copyright (c) 2025 Institute of Material Science and Sustainability (IMSS)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Individual data sources (Materials Project, etc.) are subject to their own licenses. When using the Materials Project data, please cite the appropriate references as required by their terms of use.