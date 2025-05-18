import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Element, Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.vis.structure_vtk import StructureVis
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.core.periodic_table import ElementBase
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pathlib import Path
import random

def load_synthetic_data(data_path="./data"):
    """
    Load the generated synthetic data for analysis.
    
    Args:
        data_path: Path to the data directory
    
    Returns:
        Dictionary containing all loaded datasets
    """
    # Check if the data directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory {data_path} not found. Please run the synthetic data generator first.")
    
    # Load materials summary
    materials_df = pd.read_csv(os.path.join(data_path, "materials_project/materials_summary.csv"))
    
    # Load sustainability data
    sustainability_df = pd.read_csv(os.path.join(data_path, "sustainability/element_sustainability.csv"))
    
    # Load topological candidates
    topo_candidates_df = pd.read_csv(os.path.join(data_path, "topological/top_topological_candidates.csv"))
    
    # Load combined dataset
    combined_df = pd.read_csv(os.path.join(data_path, "processed/combined_dataset.csv"))
    
    # Load best candidates
    best_candidates_df = pd.read_csv(os.path.join(data_path, "processed/top_sustainable_topological_candidates.csv"))
    
    # Try to load JARVIS data if it exists
    jarvis_dft = None
    jarvis_topo = None
    try:
        with open(os.path.join(data_path, "jarvis/jarvis_dft_3d.json"), 'r') as f:
            jarvis_dft = json.load(f)
        
        with open(os.path.join(data_path, "jarvis/jarvis_topo.json"), 'r') as f:
            jarvis_topo = json.load(f)
    except FileNotFoundError:
        print("JARVIS JSON files not found. Continuing without them.")
    
    # Return all datasets in a dictionary
    return {
        "materials": materials_df,
        "sustainability": sustainability_df,
        "topo_candidates": topo_candidates_df,
        "combined": combined_df,
        "best_candidates": best_candidates_df,
        "jarvis_dft": jarvis_dft,
        "jarvis_topo": jarvis_topo
    }

def analyze_dataset_statistics(datasets):
    """
    Analyze and print statistics about the synthetic dataset.
    
    Args:
        datasets: Dictionary of loaded datasets
    """
    materials = datasets["materials"]
    
    # Basic statistics
    print("=== Dataset Statistics ===")
    print(f"Total materials: {len(materials)}")
    print(f"Topological materials: {materials['is_topological'].sum()}")
    print(f"Non-zero band gap materials: {(materials['band_gap'] > 0.001).sum()}")
    print(f"Materials with formation energy < -2 eV: {(materials['formation_energy'] < -2).sum()}")
    
    # Topological class distribution
    print("\n=== Topological Class Distribution ===")
    topo_class_counts = materials['topological_class'].value_counts()
    for cls, count in topo_class_counts.items():
        print(f"{cls}: {count} ({count/len(materials)*100:.1f}%)")
    
    # Band gap distribution
    print("\n=== Band Gap Distribution ===")
    band_gap_bins = [0, 0.1, 0.5, 1.0, 2.0, float('inf')]
    band_gap_labels = ['0-0.1 eV', '0.1-0.5 eV', '0.5-1.0 eV', '1.0-2.0 eV', '>2.0 eV']
    band_gap_counts = pd.cut(materials['band_gap'], band_gap_bins).value_counts()
    
    for i, count in enumerate(band_gap_counts):
        print(f"{band_gap_labels[i]}: {count} ({count/len(materials)*100:.1f}%)")
    
    # Element frequency
    print("\n=== Most Common Elements ===")
    all_elements = []
    for elem_list in materials['elements']:
        all_elements.extend(elem_list.split(','))
    
    element_counts = pd.Series(all_elements).value_counts()
    for elem, count in element_counts.head(10).items():
        print(f"{elem}: {count}")
    
    # Score distributions
    print("\n=== Score Distributions ===")
    for score_type in ['topo_score', 'sustainability_score', 'stability_score', 'combined_score']:
        mean = materials[score_type].mean()
        median = materials[score_type].median()
        std = materials[score_type].std()
        print(f"{score_type}: Mean={mean:.3f}, Median={median:.3f}, Std={std:.3f}")

def plot_dataset_visualizations(datasets, output_dir="./results/data_analytics"):
    """
    Create and save visualizations of the synthetic dataset.
    
    Args:
        datasets: Dictionary of loaded datasets
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    materials = datasets["materials"]
    sustainability = datasets["sustainability"]
    best_candidates = datasets["best_candidates"]
    
    # Set up matplotlib style
    plt.style.use('ggplot')
    sns.set_context("talk")
    
    # Plot 1: Band gap distribution by topological class
    plt.figure(figsize=(12, 8))
    sns.histplot(data=materials, x='band_gap', hue='topological_class', kde=True, bins=20)
    plt.title('Band Gap Distribution by Topological Class')
    plt.xlabel('Band Gap (eV)')
    plt.ylabel('Count')
    plt.xlim(0, 3)
    plt.savefig(os.path.join(output_dir, 'band_gap_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Formation energy vs. stability
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(materials['formation_energy'], materials['e_above_hull'], 
                c=materials['is_topological'].astype(int), 
                cmap='coolwarm', 
                alpha=0.6)
    plt.title('Formation Energy vs. Energy Above Hull')
    plt.xlabel('Formation Energy (eV/atom)')
    plt.ylabel('Energy Above Hull (eV/atom)')
    
    # Create a proper colorbar with a normalization that matches the data
    cbar = plt.colorbar(scatter)
    cbar.set_label('Is Topological')
    # Set custom tick labels for the colorbar
    cbar.set_ticks([0.25, 0.75])  # Middle of each color
    cbar.set_ticklabels(['False', 'True'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'formation_vs_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Element sustainability scores
    plt.figure(figsize=(14, 10))
    
    # Sort by sustainability score
    sustainability_sorted = sustainability.sort_values('sustainability_score')
    
    # Plot elements sustainability
    plt.barh(sustainability_sorted['element'], sustainability_sorted['sustainability_score'], 
             color=plt.cm.viridis(sustainability_sorted['sustainability_score']))
    plt.title('Element Sustainability Scores')
    plt.xlabel('Sustainability Score')
    plt.ylabel('Element')
    plt.xlim(0, 1)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'element_sustainability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Score distributions
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.histplot(materials['topo_score'], kde=True, ax=axs[0, 0], color='blue')
    axs[0, 0].set_title('Topological Score Distribution')
    axs[0, 0].set_xlim(0, 1)
    
    sns.histplot(materials['sustainability_score'], kde=True, ax=axs[0, 1], color='green')
    axs[0, 1].set_title('Sustainability Score Distribution')
    axs[0, 1].set_xlim(0, 1)
    
    sns.histplot(materials['stability_score'], kde=True, ax=axs[1, 0], color='orange')
    axs[1, 0].set_title('Stability Score Distribution')
    axs[1, 0].set_xlim(0, 1)
    
    sns.histplot(materials['combined_score'], kde=True, ax=axs[1, 1], color='purple')
    axs[1, 1].set_title('Combined Score Distribution')
    axs[1, 1].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: 3D scatter plot of scores
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(materials['topo_score'], 
                         materials['sustainability_score'], 
                         materials['stability_score'],
                         c=materials['combined_score'], 
                         cmap='viridis',
                         s=50, alpha=0.6)
    
    # Highlight top candidates
    top_materials = materials[materials['combined_score'] > 0.7]
    ax.scatter(top_materials['topo_score'], 
               top_materials['sustainability_score'], 
               top_materials['stability_score'],
               color='red', s=100, alpha=1.0, label='Top Candidates')
    
    ax.set_xlabel('Topological Score')
    ax.set_ylabel('Sustainability Score')
    ax.set_zlabel('Stability Score')
    ax.set_title('Score Space - Materials Distribution')
    
    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Combined Score')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_space_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Top 20 candidates
    top20 = best_candidates.sort_values('combined_score', ascending=False).head(20)
    
    plt.figure(figsize=(14, 10))
    bars = plt.barh(top20['formula'], top20['combined_score'], color='purple')
    
    # Add score components
    for i, (_, row) in enumerate(top20.iterrows()):
        plt.barh(i, row['topo_score'] * 0.4, color='blue', alpha=0.7, height=0.3)
        plt.barh(i, row['sustainability_score'] * 0.3, left=row['topo_score'] * 0.4, 
                 color='green', alpha=0.7, height=0.3)
        plt.barh(i, row['stability_score'] * 0.3, 
                 left=row['topo_score'] * 0.4 + row['sustainability_score'] * 0.3, 
                 color='orange', alpha=0.7, height=0.3)
    
    plt.title('Top 20 Sustainable Topological Materials')
    plt.xlabel('Combined Score')
    plt.ylabel('Material Formula')
    plt.xlim(0, 1)
    plt.grid(axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Topological (40%)'),
        Patch(facecolor='green', alpha=0.7, label='Sustainability (30%)'),
        Patch(facecolor='orange', alpha=0.7, label='Stability (30%)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top20_candidates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 7: Band gap vs Topological Score
    plt.figure(figsize=(12, 8))
    plt.scatter(materials['band_gap'], materials['topo_score'], 
                c=materials['stability_score'], cmap='viridis', 
                alpha=0.6, s=50)
    plt.colorbar(label='Stability Score')
    plt.title('Band Gap vs Topological Score')
    plt.xlabel('Band Gap (eV)')
    plt.ylabel('Topological Score')
    plt.xlim(0, 2)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bandgap_vs_toposcore.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def analyze_crystal_structures(datasets, data_path="./data", sample_size=5):
    """
    Analyze a sample of crystal structures from the dataset.
    
    Args:
        datasets: Dictionary of loaded datasets
        data_path: Path to the data directory
        sample_size: Number of structures to analyze
    """
    # Get best candidates
    best_candidates = datasets["best_candidates"].sort_values('combined_score', ascending=False).head(20)
    
    # Randomly sample structures
    sample_ids = random.sample(list(best_candidates['material_id']), min(sample_size, len(best_candidates)))
    
    print("\n=== Crystal Structure Analysis ===")
    
    for material_id in sample_ids:
        poscar_path = os.path.join(data_path, f"materials_project/structures/{material_id}.vasp")
        
        try:
            # Load structure
            structure = Structure.from_file(poscar_path)
            
            # Get material data
            material_data = best_candidates[best_candidates['material_id'] == material_id].iloc[0]
            
            print(f"\nAnalyzing {material_id}: {material_data['formula']}")
            print(f"  Combined Score: {material_data['combined_score']:.3f}")
            print(f"  Topological Score: {material_data['topo_score']:.3f}")
            print(f"  Band Gap: {material_data['band_gap']:.3f} eV")
            
            # Analyze space group
            sg_analyzer = SpacegroupAnalyzer(structure)
            spacegroup = sg_analyzer.get_space_group_symbol()
            print(f"  Space Group: {spacegroup} (#{sg_analyzer.get_space_group_number()})")
            
            # Analyze composition
            comp = structure.composition
            print(f"  Composition: {comp.formula}")
            print(f"  Number of elements: {len(comp.elements)}")
            print(f"  Number of atoms: {len(structure)}")
            
            # Analyze lattice
            lattice = structure.lattice
            print(f"  Lattice parameters: a={lattice.a:.3f}, b={lattice.b:.3f}, c={lattice.c:.3f}")
            print(f"  Lattice angles: α={lattice.alpha:.1f}°, β={lattice.beta:.1f}°, γ={lattice.gamma:.1f}°")
            print(f"  Volume: {lattice.volume:.3f} Å³")
            
            # Analyze topology-relevant metrics
            elements = [str(el) for el in comp.elements]
            topo_elements = ['Bi', 'Sb', 'Te', 'Se', 'Sn', 'Pb', 'Hg', 'Tl']
            has_topo_elements = any(el in topo_elements for el in elements)
            print(f"  Contains topological elements: {has_topo_elements}")
            
            # Check for inversion symmetry
            has_inversion = sg_analyzer.have_inverse()
            print(f"  Has inversion symmetry: {has_inversion}")
            
            print(f"  Z2 invariant: {material_data['z2_invariant']}")
            
            print("  ---")
            
        except Exception as e:
            print(f"Error analyzing {material_id}: {e}")

def main():
    """Main function to analyze the synthetic dataset."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (main project directory)
    project_dir = os.path.abspath(os.path.join(script_dir, ".."))
    
    parser = argparse.ArgumentParser(description="Analyze synthetic data for topological quantum materials.")
    parser.add_argument("--data_path", default=os.path.join(project_dir, "data"), 
                        help="Path to the data directory")
    parser.add_argument("--output_dir", default=os.path.join(project_dir, "results/data_analytics"), 
                        help="Directory to save visualizations")
    parser.add_argument("--analyze_structures", action="store_true", help="Analyze crystal structures")
    parser.add_argument("--sample_size", type=int, default=5, help="Number of structures to analyze")
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data directory {args.data_path} not found.")
        print("Please run the synthetic data generator first:")
        print("./generate_synthetic_data.sh")
        sys.exit(1)
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_synthetic_data(args.data_path)
    
    # Analyze dataset statistics
    analyze_dataset_statistics(datasets)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_dataset_visualizations(datasets, args.output_dir)
    
    # Analyze crystal structures if requested
    if args.analyze_structures:
        analyze_crystal_structures(datasets, args.data_path, args.sample_size)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()