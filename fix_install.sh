#!/bin/bash

# Installation script for Topological Materials Diffusion project
# This script handles the complex dependencies, especially PyTorch Geometric

echo "Installing dependencies for Topological Materials Diffusion project..."

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Using virtual environment: $VIRTUAL_ENV"
else
    echo "Warning: Not in a virtual environment. Consider creating one with:"
    echo "python3 -m venv venv && source venv/bin/activate"
fi

# Install basic dependencies first
echo "Installing basic dependencies..."
pip install numpy>=1.20.0
pip install scipy>=1.7.0
pip install matplotlib>=3.4.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0
pip install tqdm>=4.62.0
pip install pyyaml>=6.0
pip install requests
pip install seaborn>=0.11.0

# Install PyTorch (CPU version for compatibility)
echo "Installing PyTorch..."
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric and related packages
echo "Installing PyTorch Geometric..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch-geometric

# Install materials science packages
echo "Installing materials science packages..."
pip install pymatgen>=2022.0.0
pip install ase>=3.22.0

# Install JARVIS-tools (may take some time)
echo "Installing JARVIS-tools..."
pip install jarvis-tools>=2021.07.19

# Install additional dependencies
echo "Installing additional dependencies..."
pip install networkx>=2.6.0
pip install plotly>=5.3.0
pip install tensorboard>=2.8.0

# Try to install optional dependencies
echo "Installing optional dependencies..."
pip install wandb>=0.12.0 || echo "Warning: Could not install wandb"
pip install e3nn>=0.5.0 || echo "Warning: Could not install e3nn"
pip install dgl>=0.8.0 || echo "Warning: Could not install dgl"

# Install testing dependencies
echo "Installing testing dependencies..."
pip install pytest>=6.2.5
pip install jupyter>=1.0.0

echo "Installation complete!"
echo ""
echo "To verify the installation, run:"
echo "python -c \"import torch; import torch_geometric; import pymatgen; import jarvis; print('All major dependencies installed successfully!')\""
