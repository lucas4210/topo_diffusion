# Running Instructions for Topological Materials Diffusion Project

This document provides detailed instructions for running the refactored codebase for the Generative Diffusion Model for Sustainable Topological Quantum Materials project.

## Setup Environment

First, set up your Python environment:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

The project has a simplified structure:

```
topo_diffusion_refactored/
├── src/                # Core source code
│   ├── __init__.py     # Package initialization
│   ├── data.py         # Data processing and crystal graph representation
│   ├── model.py        # Diffusion model architecture
│   ├── training.py     # Training pipeline and validation tools
│   ├── utils.py        # Utility functions
│   └── main.py         # Command-line interface and main scripts
├── config.yaml         # Configuration file
└── requirements.txt    # Dependencies
```

## Command-Line Interface

The project provides a command-line interface for all operations:

### 1. Download Data

Download and process JARVIS datasets:

```bash
python -m src.main download --output-dir data/raw --limit 1000
```

Options:
- `--output-dir`: Directory to store downloaded data (default: "data/raw")
- `--limit`: Limit the number of structures to download (default: all)
- `--log-level`: Set the logging level (default: "INFO")

### 2. Train Model

Train the diffusion model:

```bash
python -m src.main train --config config.yaml --data-dir data/processed --checkpoint-dir checkpoints
```

Options:
- `--config`: Path to the configuration file (required)
- `--data-dir`: Directory containing processed data (default: "data/processed")
- `--checkpoint-dir`: Directory to save checkpoints (default: "checkpoints")
- `--device`: Device to train on (cuda or cpu) (default: "cuda" if available, else "cpu")
- `--use-wandb`: Whether to use Weights & Biases for logging (flag)
- `--log-level`: Set the logging level (default: "INFO")

### 3. Generate Materials

Generate new materials using the trained model:

```bash
python -m src.main generate --model-checkpoint checkpoints/best_model.pt --num-samples 10 --output-dir generated_materials
```

Options:
- `--model-checkpoint`: Path to the model checkpoint (required)
- `--num-samples`: Number of materials to generate (default: 10)
- `--batch-size`: Batch size for generation (default: 4)
- `--num-nodes`: Number of nodes in each generated graph (default: 16)
- `--condition`: Conditioning parameters (format: 'key1=value1,key2=value2') (default: None)
- `--output-dir`: Directory to save generated materials (default: "generated_materials")
- `--device`: Device to generate on (cuda or cpu) (default: "cuda" if available, else "cpu")
- `--log-level`: Set the logging level (default: "INFO")

### 4. Validate Materials

Validate generated materials:

```bash
python -m src.main validate --input-dir generated_materials --output-dir validation_results
```

Options:
- `--input-dir`: Directory containing materials to validate (required)
- `--output-dir`: Directory to save validation results (default: "validation_results")
- `--run-dft`: Whether to run DFT validation (flag)
- `--log-level`: Set the logging level (default: "INFO")

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download data
python -m src.main download --output-dir data/raw --limit 500

# 3. Train model
python -m src.main train --config config.yaml --checkpoint-dir checkpoints

# 4. Generate materials
python -m src.main generate --model-checkpoint checkpoints/best_model.pt --num-samples 20 --output-dir generated_materials

# 5. Validate materials
python -m src.main validate --input-dir generated_materials --output-dir validation_results
```

## Using as a Python Package

You can also use the codebase as a Python package in your own scripts:

```python
from src.data import JARVISDataDownloader, CrystalGraphDataset
from src.model import CrystalGraphDiffusionModel, DiffusionProcess
from src.training import DiffusionTrainer, MaterialValidator
from src.utils import visualize_structure, calculate_sustainability_metrics

# Example: Download data
downloader = JARVISDataDownloader(data_dir="data/raw")
dft_path = downloader.download_dft_data(limit=100)

# Example: Create and train model
# ... (see documentation for details)

# Example: Generate and validate materials
# ... (see documentation for details)
```

## Configuration

The `config.yaml` file contains all configuration parameters for the model and training process. You can modify this file to adjust hyperparameters, model architecture, and training settings.

## Outputs

- Downloaded data is saved in the specified output directory
- Model checkpoints are saved in the checkpoint directory
- Generated materials are saved as CIF files in the output directory
- Validation results include JSON data and HTML reports

## Troubleshooting

If you encounter any issues:

1. Check the log files in the respective output directories
2. Ensure all dependencies are installed correctly
3. Verify that the data paths are correct
4. For CUDA errors, try running on CPU with `--device cpu`
