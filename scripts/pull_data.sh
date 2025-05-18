#!/bin/bash
# Script to download JARVIS data components for a comprehensive database solution

# Create directory for JARVIS data
mkdir -p data/jarvis

# Download JARVIS-DFT for materials structures and properties
echo "Downloading JARVIS-DFT data..."
wget https://jarvis.nist.gov/static/jarvis_dft_3d.json.gz -O data/jarvis/jarvis_dft_3d.json.gz
gunzip data/jarvis/jarvis_dft_3d.json.gz

# Download JARVIS-TOPO for topological classifications
echo "Downloading JARVIS-TOPO data..."
wget https://jarvis.nist.gov/static/jarvis_topo.json.gz -O data/jarvis/jarvis_topo.json.gz
gunzip data/jarvis/jarvis_topo.json.gz

# Download JARVIS-FF for additional properties (optional)
echo "Downloading JARVIS-FF data..."
wget https://jarvis.nist.gov/static/jarvis_ff_qm9.json.gz -O data/jarvis/jarvis_ff_qm9.json.gz
gunzip data/jarvis/jarvis_ff_qm9.json.gz

echo "All JARVIS data components downloaded successfully."

# Run the Python script to process the data
echo "Running the Python script to process the data..."
python3 /scripts/pull_data.py

echo "Data processing complete!"