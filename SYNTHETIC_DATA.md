# Synthetic Data Generation for Topological Quantum Materials

This document provides an explanation of the synthetic data generation approach designed for the Sustainable Topological Quantum Materials project. This synthetic data generation solution is useful when the JARVIS database is unavailable.

## Overview

The provided scripts generate synthetic crystal structures and properties that mimic the expected data from the JARVIS database, including:

1. Crystal structures with realistic lattice parameters and atomic positions
2. Material properties including band gaps, formation energies, and topological classifications
3. Sustainability metrics for elements
4. Directory structure and file formats compatible with the existing codebase

## Quick Start

To generate the synthetic data:

```bash
# Make the script executable
chmod +x generate_synthetic_data.sh

# Run the script
./generate_synthetic_data.sh
```

The script will create a `data` directory with a structure matching what would be downloaded from the JARVIS database.

## Generated Data Structure

The generated data will have the following structure:

```
data/
├── jarvis/                         # JARVIS-like data
│   ├── jarvis_dft_3d.json          # Synthetic DFT data
│   └── jarvis_topo.json            # Synthetic topological classifications
│
├── materials_project/              # Processed materials data
│   ├── materials_summary.csv       # Summary of all materials properties
│   ├── training_subset.csv         # Subset for training
│   ├── structures/                 # POSCAR files for all materials
│   └── training_subset_structures/ # POSCAR files for training subset
│
├── sustainability/                 # Sustainability metrics
│   └── element_sustainability.csv  # Element-level sustainability scores
│
├── topological/                    # Topological property data
│   ├── topological_predictions.csv # All materials with topo predictions
│   ├── top_topological_candidates.csv # High-scoring candidates
│   └── structures/                 # POSCAR files for top candidates
│
└── processed/                      # Combined datasets
    ├── combined_dataset.csv        # Complete dataset with all properties
    └── top_sustainable_topological_candidates.csv # Best candidates
```

## Data Generation Approach

### Crystal Structures

The synthetic data generator creates:

- Random crystal structures with 1-4 elements
- Reasonable lattice parameters and atomic positions
- Biased selection of elements and space groups known to host topological phases
- POSCAR files compatible with pymatgen

### Material Properties

For each structure, synthetic properties are generated including:

- Band gaps (influenced by topological classification)
- Formation energies and stability metrics
- Topological classifications and Z2 invariants
- Density and other physical properties

### Sustainability Metrics

For each element, the generator creates:

- Abundance scores (based on natural abundance patterns)
- Toxicity ratings (higher for toxic elements like Hg, Pb, etc.)
- Recyclability scores
- Overall sustainability scores

### Combined Metrics

The generator also calculates combined scores:

- Topological scores based on band gap and element composition
- Stability scores based on energy above hull
- Overall scores balancing topological, stability, and sustainability concerns

## Configuration Options

The generator has several configuration options in the script:

- `num_materials`: Total number of materials to generate (default: 5000)
- `num_topological`: Number of topological materials (default: 800)
- `training_subset_size`: Size of training subset (default: 2000)
- `random_seed`: For reproducibility (default: 42)
- Lists of elements, topological elements, and space groups

These can be adjusted in the Python script as needed.

## Data Characteristics

The synthetic data mimics key patterns found in real topological materials:

1. **Topological materials** tend to:
   - Have smaller band gaps (< 0.3 eV)
   - Include elements like Bi, Sb, Te, Se, etc.
   - Occur in certain space groups
   - Have specific Z2 invariants

2. **Sustainability patterns**:
   - Common elements have higher abundance scores
   - Toxic elements have higher toxicity ratings
   - Recyclable elements have higher recyclability scores

3. **Structure-property relationships**:
   - Complex materials tend to have lower formation energies
   - Larger unit cells affect physical properties
   - Element combinations influence stability

## Usage with Existing Code

The synthetic data is designed to be compatible with the existing codebase. You can:

1. Use the generated POSCAR files with `CrystalGraphConverter` class
2. Train the diffusion model using the synthetic properties
3. Evaluate the model using the synthetic test set

The data format matches what's described in the DATASET.md file and expected by the various Python modules.

## Limitations

The synthetic data has some limitations compared to real JARVIS data:

- Less accurate structure-property relationships
- Simplified band structure and topological classifications
- Less nuanced element distribution patterns
- Limited physical consistency (some structures may be unrealistic)

However, it provides a suitable substitute for developing and testing the model architecture before acquiring real data.

## Future Improvements

Future versions of the synthetic data generator could include:

- More sophisticated crystal structure generation based on space group symmetry
- DFT-informed property correlations
- Improved topological classification based on symmetry indicators
- More accurate sustainability metrics from real-world data