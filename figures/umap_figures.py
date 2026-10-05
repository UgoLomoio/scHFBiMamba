"""
UMAP projections of latent space embeddings and classifier features.

For each of 4 models × 2 embedding types × 3 colorings = 24 figures.
  - Embedding types: autoencoder latent (350-dim), classifier features (128 or 105-dim)
  - Colorings: disease state, species, cell type

Uses umap-learn for dimensionality reduction on the test set.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from config import (
    ABLATION_MODELS, DISEASE_CLASSES, SPECIES_CLASSES, CELLTYPE_CLASSES,
    UMAPConfig
)
from data import parse_label

FIGURE_DPI = 300

# Colorblind-friendly palettes
DISEASE_COLORS = {
    'CTRL': '#75A025', 'HFpEF': '#0279EE', 'HFrEF': '#FF9400', 'AS': '#FD9BED',
}
SPECIES_COLORS = {
    'Human': '#0279EE', 'Mice': '#FF9400',
}
CELLTYPE_COLORS = {
    'Cardiomyocytes': '#0279EE', 'Endothelial': '#FF9400', 'Fibroblasts': '#75A025',
    'Immune.cells': '#FD9BED', 'Neuro': '#E9ED4C', 'Pericytes': '#FF6B6B',
    'Smooth.Muscle': '#9B59B6',
}


def compute_umap(embeddings: np.ndarray, config: UMAPConfig = None) -> np.ndarray:
    """
    Compute UMAP projection of embeddings.

    Args:
        embeddings: (N, D) high-dimensional embeddings
        config: UMAPConfig with hyperparameters

    Returns:
        (N, 2) 2D UMAP coordinates
    """
    import umap

    if config is None:
        config = UMAPConfig()

    reducer = umap.UMAP(
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric=config.metric,
        n_components=config.n_components,
        random_state=config.random_state,
        verbose=True,
    )

    coords = reducer.fit_transform(embeddings)
    return coords


def plot_umap(
    coords: np.ndarray,
    labels_str: List[str],
    color_by: str,
    title: str,
    output_dir: str,
    name: str,
):
    """
    Plot UMAP coordinates colored by a label category.

    Args:
        coords: (N, 2) UMAP coordinates
        labels_str: list of label strings (e.g. "Human-HFpEF_Cardiomyocytes")
        color_by: "disease", "species", or "celltype"
        title: figure title
        output_dir: where to save
        name: file name prefix
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Extract the relevant label component
    if color_by == "disease":
        values = [parse_label(l)[1] for l in labels_str]
        color_map = DISEASE_COLORS
    elif color_by == "species":
        values = [parse_label(l)[0] for l in labels_str]
        color_map = SPECIES_COLORS
    else:  # celltype
        values = [parse_label(l)[2] for l in labels_str]
        color_map = CELLTYPE_COLORS

    # Plot each group
    for val in color_map:
        mask = [v == val for v in values]
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        ax.scatter(
            coords[indices, 0], coords[indices, 1],
            c=color_map[val], s=3, alpha=0.5, label=val, edgecolors='none'
        )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title, fontweight='bold')
    ax.legend(markerscale=3, fontsize=8, loc='best')

    for fmt in ['svg', 'png']:
        path = os.path.join(output_dir, f'{name}.{fmt}')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}.svg/png")


def run_umap_for_model(
    model_id: str,
    model_dir: str,
    test_labels_str: List[str],
    output_dir: str,
    umap_config: UMAPConfig = None,
):
    """
    Compute and plot UMAP for a single model (both latent and classifier features).

    Args:
        model_id: model identifier (M1-M4)
        model_dir: directory containing test_latent_embeddings.npy and test_clf_features.npy
        test_labels_str: list of label strings for test cells
        output_dir: where to save figures
        umap_config: UMAP hyperparameters
    """
    os.makedirs(output_dir, exist_ok=True)

    embedding_types = [
        ('latent', 'test_latent_embeddings.npy', 'Autoencoder Latent'),
        ('features', 'test_clf_features.npy', 'Classifier Features'),
    ]

    for emb_type, filename, emb_label in embedding_types:
        emb_path = os.path.join(model_dir, filename)
        if not os.path.exists(emb_path):
            print(f"  Warning: {emb_path} not found, skipping {model_id}/{emb_type}")
            continue

        embeddings = np.load(emb_path)
        print(f"  {model_id} {emb_type}: {embeddings.shape}")

        # Compute UMAP
        coords = compute_umap(embeddings, umap_config)

        # Save coordinates
        coords_path = os.path.join(output_dir, f'{model_id}_{emb_type}_umap_coords.npy')
        np.save(coords_path, coords)

        # Plot with 3 colorings
        for color_by in ['disease', 'species', 'celltype']:
            name = f'{model_id}_{emb_type}_{color_by}'
            title = f'{model_id} {emb_label} — colored by {color_by}'
            plot_umap(coords, test_labels_str, color_by, title, output_dir, name)


def run_all_umap(
    ablation_dir: str,
    output_dir: str,
    test_labels_path: str,
    umap_config: UMAPConfig = None,
    model_ids: List[str] = None,
):
    """
    Run UMAP for all models and save all figures.

    Args:
        ablation_dir: directory with model subdirectories
        output_dir: where to save UMAP figures
        test_labels_path: path to test label CSV
        umap_config: UMAP hyperparameters
        model_ids: list of model IDs (default: all)
    """
    print("\n=== UMAP Projections ===")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'umap_coords'), exist_ok=True)

    if model_ids is None:
        model_ids = list(ABLATION_MODELS.keys())

    # Load test labels
    from data import load_label_strings
    test_labels_str = load_label_strings(test_labels_path)

    for model_id in model_ids:
        model_dir = os.path.join(ablation_dir, model_id)
        if not os.path.isdir(model_dir):
            print(f"  Warning: {model_dir} not found, skipping {model_id}")
            continue

        print(f"\n  Processing {model_id}...")
        run_umap_for_model(
            model_id, model_dir, test_labels_str,
            output_dir, umap_config
        )

    print(f"\nUMAP figures saved to {output_dir}/")
