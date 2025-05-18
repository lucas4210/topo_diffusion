#!/bin/bash
# Script to generate synthetic data for topological quantum materials research
# Use this when JARVIS database is unavailable

# Get the directory where this script is located (scripts directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the parent directory (main project directory)
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Create directory for data in the main project directory
mkdir -p "$PROJECT_DIR/data"

# Print banner
echo "=================================================="
echo "   Synthetic Data Generator for TQM-GML Project   "
echo "=================================================="
echo ""
echo "This script will generate synthetic data that mimics the JARVIS database"
echo "structure and content, suitable for developing the diffusion model."
echo ""
echo "Scripts location: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Data will be generated in: $PROJECT_DIR/data"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found on your system."
    exit 1
fi

# Check for required Python packages
echo "Checking for required Python packages..."
python3 -c "import numpy, pandas, tqdm, json" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install numpy pandas tqdm
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install required packages. Please install them manually:"
        echo "pip install numpy pandas tqdm"
        exit 1
    fi
fi

# Check if the Python script exists in the scripts directory
if [ ! -f "$SCRIPT_DIR/synthetic_data_generator.py" ]; then
    echo "Error: synthetic_data_generator.py not found in $SCRIPT_DIR"
    echo "Make sure synthetic_data_generator.py is in the scripts directory."
    exit 1
fi

# Run the Python script to generate synthetic data in the main project directory
echo "Generating synthetic data..."
cd "$PROJECT_DIR"  # Change to the project directory
python3 "$SCRIPT_DIR/synthetic_data_generator.py" --output_dir="$PROJECT_DIR/data"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "   Synthetic Data Generation Complete!            "
    echo "=================================================="
    echo ""
    echo "The synthetic data has been generated successfully."
    echo "You can now proceed with model development using the"
    echo "generated data as a substitute for JARVIS database."
    echo ""
    echo "Data location: $PROJECT_DIR/data/"
    echo ""
else
    echo "Error: Data generation failed."
    exit 1
fi