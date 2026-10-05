"""
Per-label explainability figures for HF Mamba ablation models.

For each of the 13 labels × 4 models = 52 beeswarm figures.
Each figure shows the top-20 genes by mean |SHAP| for that label,
with SHAP values on the x-axis and expression levels as color.

All models use gene-level SHAP (via composite model), so attributions
are at the 16,545 gene level for all models.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from config import (
    ALL_CLASSES, N_CLASSES, N_GENES, ABLATION_MODELS,
    SPECIES_CLASSES, CELLTYPE_CLASSES, DISEASE_CLASSES,
    SPECIES_IDX, CELLTYPE_IDX, DISEASE_IDX,
    AttributionConfig
)
from data import parse_label

FIGURE_DPI = 300


def sample_cells_for_label(
    test_labels_str: List[str],
    label_name: str,
    label_type: str,
    n_cells: int = 100,
    seed: int = 1111,
) -> np.ndarray:
    """
    Sample random cell indices that have a specific label.

    Args:
        test_labels_str: list of label strings
        label_name: the label to match (e.g. "Human", "Cardiomyocytes", "HFpEF")
        label_type: "species", "celltype", or "disease"
        n_cells: number of cells to sample
        seed: random seed

    Returns:
        array of cell indices
    """
    rng = np.random.RandomState(seed)

    # Find matching cells
    matching = []
    for i, label_str in enumerate(test_labels_str):
        species, disease, cell_type = parse_label(label_str)
        if label_type == "species" and species == label_name:
            matching.append(i)
        elif label_type == "celltype" and cell_type == label_name:
            matching.append(i)
        elif label_type == "disease" and disease == label_name:
            matching.append(i)

    if len(matching) == 0:
        return np.array([], dtype=int)

    n_sample = min(n_cells, len(matching))
    selected = rng.choice(matching, size=n_sample, replace=False)
    return np.sort(selected)


def get_label_type_and_index(label_name: str) -> Tuple[str, int]:
    """
    Determine the type and index of a label.

    Args:
        label_name: one of the 13 class names

    Returns:
        (label_type, class_index) where label_type is "species", "celltype", or "disease"
    """
    if label_name in SPECIES_CLASSES:
        return "species", SPECIES_CLASSES.index(label_name)
    elif label_name in CELLTYPE_CLASSES:
        return "celltype", 2 + CELLTYPE_CLASSES.index(label_name)
    elif label_name in DISEASE_CLASSES:
        return "disease", 9 + DISEASE_CLASSES.index(label_name)
    else:
        raise ValueError(f"Unknown label: {label_name}")


def plot_beeswarm(
    shap_values: np.ndarray,
    expression: np.ndarray,
    gene_names: List[str],
    label_name: str,
    model_id: str,
    output_dir: str,
    top_n: int = 20,
):
    """
    Plot a beeswarm figure for a single label.

    Args:
        shap_values: (N_cells, N_genes) SHAP values for this label's class
        expression: (N_cells, N_genes) expression values for the sampled cells
        gene_names: list of gene names
        label_name: name of the label being explained
        model_id: model identifier
        output_dir: where to save
        top_n: number of top genes to show
    """
    # Mean absolute SHAP per gene
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(7, 8))

    for rank, g_idx in enumerate(top_indices):
        shap_vals = shap_values[:, g_idx]
        expr_vals = expression[:, g_idx]

        # Jitter y positions
        y_jitter = np.random.normal(rank, 0.12, size=len(shap_vals))

        # Color by expression (data-driven colorbar limits)
        expr_min = float(np.min(expr_vals))
        expr_max = float(np.max(expr_vals))
        if expr_max <= expr_min:
            expr_max = expr_min + 1e-6
        scatter = ax.scatter(
            shap_vals, y_jitter, c=expr_vals, cmap='coolwarm',
            s=12, alpha=0.6, edgecolors='none', vmin=expr_min, vmax=expr_max
        )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([gene_names[i] for i in top_indices], fontsize=8)
    ax.set_xlabel('SHAP value')
    ax.set_title(f'{model_id}: Top {top_n} predictors for {label_name}',
                 fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Expression')

    plt.tight_layout()
    for fmt in ['svg', 'png']:
        path = os.path.join(output_dir, f'{model_id}_label_{label_name}.{fmt}')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def run_explainability_for_model(
    model_id: str,
    model_dir: str,
    test_features: np.ndarray,
    test_labels_str: List[str],
    gene_names: List[str],
    output_dir: str,
    config: AttributionConfig = None,
):
    """
    Generate per-label beeswarm figures for a single model.

    Uses pre-computed SHAP values from the attribution directory.
    For each of the 13 labels, samples 100 cells with that label and
    plots the top-20 genes by mean |SHAP|.

    Args:
        model_id: model identifier
        model_dir: directory with attribution/ subdirectory
        test_features: (N, N_genes) test expression
        test_labels_str: list of label strings
        gene_names: list of gene names
        output_dir: where to save figures
        config: attribution config
    """
    if config is None:
        config = AttributionConfig()

    os.makedirs(output_dir, exist_ok=True)

    # Load SHAP values
    shap_path = os.path.join(model_dir, 'attribution', f'{model_id}_shap_values.npy')
    if not os.path.exists(shap_path):
        print(f"  Warning: {shap_path} not found, skipping {model_id}")
        return

    shap_values = np.load(shap_path)  # (N_classes, N_subset, N_genes)
    print(f"  {model_id} SHAP values: {shap_values.shape}")

    # Load attribution labels to map SHAP rows to test cells
    attrib_labels_path = os.path.join(model_dir, 'attribution', f'{model_id}_shap_test_labels.txt')
    with open(attrib_labels_path, 'r') as f:
        attrib_labels_str = f.read().strip().split('\n')

    # For each of the 13 labels
    for label_name in ALL_CLASSES:
        label_type, class_idx = get_label_type_and_index(label_name)

        # Sample cells with this label from the attribution subset
        cell_indices = sample_cells_for_label(
            attrib_labels_str, label_name, label_type,
            n_cells=config.n_cells_per_label,
            seed=config.explainability_seed,
        )

        if len(cell_indices) == 0:
            print(f"    No cells for label {label_name}, skipping")
            continue

        # Get SHAP values for this class and sampled cells
        class_shap = shap_values[class_idx, cell_indices, :]  # (n_cells, N_genes)

        # Get expression for these cells from the attribution subset
        # The attribution subset is a balanced sample of the test set
        # We need to get the corresponding expression values
        # Since we don't have direct mapping, use the attribution labels
        # to find matching cells in the test set
        from data import sample_balanced_test_subset
        subset_features, _, subset_labels_str = sample_balanced_test_subset(
            test_features, test_labels_str,
            n_per_celltype=200, seed=1111
        )

        cell_expression = np.asarray(subset_features[cell_indices], dtype=np.float32)

        print(f"    {label_name}: {len(cell_indices)} cells, "
              f"top-{config.top_n_genes} genes")

        plot_beeswarm(
            class_shap, cell_expression, gene_names,
            label_name, model_id, output_dir,
            top_n=config.top_n_genes,
        )


def run_all_explainability(
    ablation_dir: str,
    output_dir: str,
    test_features: np.ndarray,
    test_labels_str: List[str],
    gene_names: List[str],
    config: AttributionConfig = None,
    model_ids: List[str] = None,
):
    """
    Generate per-label beeswarm figures for all models.

    Args:
        ablation_dir: directory with model subdirectories
        output_dir: where to save figures
        test_features: (N, N_genes) test expression
        test_labels_str: list of label strings
        gene_names: list of gene names
        config: attribution config
        model_ids: list of model IDs (default: all)
    """
    print("\n=== Per-Label Explainability Figures ===")
    os.makedirs(output_dir, exist_ok=True)

    if model_ids is None:
        model_ids = list(ABLATION_MODELS.keys())

    for model_id in model_ids:
        model_dir = os.path.join(ablation_dir, model_id)
        if not os.path.isdir(model_dir):
            print(f"  Warning: {model_dir} not found, skipping {model_id}")
            continue

        print(f"\n  Processing {model_id}...")
        run_explainability_for_model(
            model_id, model_dir,
            test_features, test_labels_str, gene_names,
            output_dir, config
        )

    print(f"\nExplainability figures saved to {output_dir}/")
