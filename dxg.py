"""
DXG (Differentially Explained Genes) analysis.

Replicates the paper's approach of identifying dysregulated genes using
XAI attribution scores (SHAP or IG) instead of raw expression values.

Pipeline (matching paper's simpleclass function):
  1. Z-transform attribution scores per cell
  2. Define comparison groups (e.g., Human CM HFpEF vs Human CM CTRL)
  3. Statistical test (t-test or Wilcoxon) on Z-transformed scores per gene
  4. Benjamini-Hochberg FDR correction
  5. Cube-root fold change: FC³ = ∛(mean(SHAP_A) - mean(SHAP_B))
  6. Filter by p_adj, SHAP threshold, FC threshold, expression threshold
  7. Classify predictors as Direct or Inverse
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ranksums, shapiro
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from config import (
    SPECIES_CLASSES, CELLTYPE_CLASSES, DISEASE_CLASSES,
    ALL_CLASSES, N_CLASSES
)
from data import parse_label, label_to_multihot


# ─── Z-transformation ────────────────────────────────────────────────────────

def z_transform_per_cell(attribution_values: np.ndarray) -> np.ndarray:
    """
    Z-transform attribution scores per cell (row-wise).
    Matches paper's z_score_loop_per_cell.

    Z_i = (x_i - μ_i) / σ_i
    where μ_i, σ_i are mean/std of attributions across all genes for cell i.

    Args:
        attribution_values: (N_cells, N_genes) attribution scores for one class

    Returns:
        (N_cells, N_genes) Z-transformed scores
    """
    mean = attribution_values.mean(axis=1, keepdims=True)
    std = attribution_values.std(axis=1, keepdims=True)
    # Avoid division by zero
    std = np.where(std == 0, 1e-108, std)
    z = (attribution_values - mean) / std
    return z


def z_transform_all_classes(shap_values: np.ndarray) -> np.ndarray:
    """
    Z-transform attribution scores per cell for all classes.
    Matches paper: each class's SHAP values are Z-transformed independently.

    Args:
        shap_values: (N_classes, N_cells, N_genes)

    Returns:
        (N_classes, N_cells, N_genes) Z-transformed
    """
    n_classes, n_cells, n_genes = shap_values.shape
    z_values = np.zeros_like(shap_values)
    for c in range(n_classes):
        z_values[c] = z_transform_per_cell(shap_values[c])
    return z_values


# ─── Group definition ────────────────────────────────────────────────────────

def define_groups(test_labels_str: List[str],
                  species: str, cell_type: str, disease: str) -> np.ndarray:
    """
    Get indices of cells matching a specific species/cell_type/disease.

    Args:
        test_labels_str: list of label strings
        species: e.g. "Human"
        cell_type: e.g. "Cardiomyocytes"
        disease: e.g. "HFpEF"

    Returns:
        boolean mask array (N_cells,)
    """
    mask = np.zeros(len(test_labels_str), dtype=bool)
    for i, label_str in enumerate(test_labels_str):
        sp, dis, ct = parse_label(label_str)
        if sp == species and ct == cell_type and dis == disease:
            mask[i] = True
    return mask


def get_all_comparisons() -> List[Dict]:
    """
    Generate all comparison pairs for DXG analysis.
    Matches paper: for each species × cell_type, compare each disease vs CTRL.

    Returns:
        list of dicts with keys: species, cell_type, disease_a, disease_b, name
    """
    comparisons = []
    for species in SPECIES_CLASSES:
        for cell_type in CELLTYPE_CLASSES:
            for disease in ['AS', 'HFpEF', 'HFrEF']:
                comparisons.append({
                    'species': species,
                    'cell_type': cell_type,
                    'disease_a': disease,
                    'disease_b': 'CTRL',
                    'name': f"{species}_{cell_type}_{disease}_vs_{species}_{cell_type}_CTRL",
                    'shap_class': 9 + DISEASE_CLASSES.index(disease),  # disease output node
                })
    return comparisons


# ─── DXG analysis ────────────────────────────────────────────────────────────

def dxg_analysis(
    z_shap_values: np.ndarray,
    test_labels_str: List[str],
    gene_names: List[str],
    test_expression: np.ndarray,
    comparison: Dict,
    testing_method: str = "t",
    shap_thresh: float = 0.01,
    logfc_thresh: float = 0.1,
    expression_threshold: float = 0.1,
    p_adj_thresh: float = 0.05,
) -> pd.DataFrame:
    """
    Perform DXG analysis for a single comparison.
    Replicates paper's simpleclass function.

    Args:
        z_shap_values: (N_classes, N_cells, N_genes) Z-transformed attribution scores
        test_labels_str: list of label strings for test cells
        gene_names: list of gene names
        test_expression: (N_cells, N_genes) raw expression values (for filtering)
        comparison: dict with species, cell_type, disease_a, disease_b, shap_class
        testing_method: "t" for t-test, "wilcoxon" for Wilcoxon rank-sum
        shap_thresh: minimum |mean_SHAP| in at least one group
        logfc_thresh: minimum |cube_root_FC|
        expression_threshold: minimum mean expression in at least one group
        p_adj_thresh: adjusted p-value threshold

    Returns:
        DataFrame with DXG results
    """
    shap_class = comparison['shap_class']
    species = comparison['species']
    cell_type = comparison['cell_type']
    disease_a = comparison['disease_a']
    disease_b = comparison['disease_b']

    # Get SHAP values for the relevant disease class
    class_shap = z_shap_values[shap_class]  # (N_cells, N_genes)

    # Define groups
    mask_a = define_groups(test_labels_str, species, cell_type, disease_a)
    mask_b = define_groups(test_labels_str, species, cell_type, disease_b)

    if mask_a.sum() < 2 or mask_b.sum() < 2:
        # Not enough cells for comparison
        return pd.DataFrame()

    shap_a = class_shap[mask_a]  # (n_a, N_genes)
    shap_b = class_shap[mask_b]  # (n_b, N_genes)

    expr_a = test_expression[mask_a]
    expr_b = test_expression[mask_b]

    # Statistical test per gene
    n_genes = class_shap.shape[1]
    results = []

    for g in range(n_genes):
        subset_a = shap_a[:, g]
        subset_b = shap_b[:, g]

        if testing_method == "t":
            stat, p_val = ttest_ind(subset_a, subset_b, equal_var=True)
        else:  # wilcoxon
            stat, p_val = ranksums(subset_a, subset_b)

        results.append({
            'Feature': gene_names[g],
            'p_val': p_val,
            'mean_SHAP_group1': np.mean(subset_a),
            'mean_SHAP_group2': np.mean(subset_b),
        })

    results_df = pd.DataFrame(results)

    # Benjamini-Hochberg correction
    rejected, p_adjusted, _, _ = multipletests(
        results_df['p_val'].values,
        alpha=0.05, method='fdr_bh', is_sorted=False
    )
    results_df['p_adj'] = p_adjusted

    # Cube-root fold change (handles negative/zero values)
    results_df['avg_cube_root_FC'] = np.cbrt(
        results_df['mean_SHAP_group1'] - results_df['mean_SHAP_group2']
    )

    # Filtering (matching paper)
    results_df = results_df[results_df['p_adj'] <= p_adj_thresh]
    results_df = results_df[
        (np.abs(results_df['mean_SHAP_group1']) >= shap_thresh) |
        (np.abs(results_df['mean_SHAP_group2']) >= shap_thresh)
    ]
    results_df = results_df[
        np.abs(results_df['avg_cube_root_FC']) >= logfc_thresh
    ]

    # Expression filter
    results_df = _expression_filter(
        results_df, gene_names, expr_a, expr_b, expression_threshold)

    # Predictor type (Direct vs Inverse)
    results_df = _predictor_type(results_df, gene_names, shap_a, expr_a)

    # Sort by fold change
    results_df = results_df.sort_values('avg_cube_root_FC', ascending=False)
    results_df = results_df.dropna()

    return results_df


def _expression_filter(results_df: pd.DataFrame, gene_names: List[str],
                       expr_a: np.ndarray, expr_b: np.ndarray,
                       threshold: float) -> pd.DataFrame:
    """Filter genes by minimum expression in at least one group."""
    features_to_remove = []
    avg_expr_1 = []
    avg_expr_2 = []

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    for feature in results_df['Feature']:
        g_idx = gene_to_idx.get(feature)
        if g_idx is None:
            features_to_remove.append(feature)
            avg_expr_1.append(0.0)
            avg_expr_2.append(0.0)
            continue

        mean_1 = np.mean(expr_a[:, g_idx])
        mean_2 = np.mean(expr_b[:, g_idx])
        avg_expr_1.append(mean_1)
        avg_expr_2.append(mean_2)

        if mean_1 < threshold and mean_2 < threshold:
            features_to_remove.append(feature)

    results_df['avg_expr_group1_testdata'] = avg_expr_1
    results_df['avg_expr_group2_testdata'] = avg_expr_2

    return results_df[~results_df['Feature'].isin(features_to_remove)]


def _predictor_type(results_df: pd.DataFrame, gene_names: List[str],
                    shap_a: np.ndarray, expr_a: np.ndarray) -> pd.DataFrame:
    """
    Classify each gene as Direct or Inverse predictor.
    Direct: positive SHAP correlates with higher expression
    Inverse: negative SHAP correlates with higher expression
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    predictor_types = []

    for feature in results_df['Feature']:
        g_idx = gene_to_idx.get(feature)
        if g_idx is None:
            predictor_types.append("NULL")
            continue

        shap_feature = shap_a[:, g_idx]
        expr_feature = expr_a[:, g_idx]

        # Separate by positive and negative SHAP values
        pos_mask = shap_feature > 0
        neg_mask = shap_feature < 0

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            mean_expr_pos = np.mean(expr_feature[pos_mask])
            mean_expr_neg = np.mean(expr_feature[neg_mask])
            ptype = "Direct" if mean_expr_pos > mean_expr_neg else "Inverse"
        else:
            ptype = "Direct" if np.mean(shap_feature) > 0 else "Inverse"

        predictor_types.append(ptype)

    results_df['predictor_type'] = predictor_types
    return results_df


# ─── Run all DXG comparisons ─────────────────────────────────────────────────

def run_all_dxg(
    shap_values: np.ndarray,
    test_labels_str: List[str],
    gene_names: List[str],
    test_expression: np.ndarray,
    output_dir: str,
    model_name: str = "",
    testing_method: str = "t",
    shap_thresh: float = 0.01,
    logfc_thresh: float = 0.1,
    expression_threshold: float = 0.1,
    p_adj_thresh: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """
    Run DXG analysis for all comparison pairs.

    Args:
        shap_values: (N_classes, N_cells, N_genes) raw attribution scores
        test_labels_str: label strings for test cells
        gene_names: gene names
        test_expression: (N_cells, N_genes) raw expression
        output_dir: where to save results
        model_name: prefix for output files

    Returns:
        dict mapping comparison name -> results DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{model_name}_" if model_name else ""

    # Z-transform
    print("Z-transforming attribution scores...")
    z_values = z_transform_all_classes(shap_values)

    # Run all comparisons
    comparisons = get_all_comparisons()
    all_results = {}

    for comp in tqdm(comparisons, desc="DXG comparisons"):
        name = comp['name']
        result = dxg_analysis(
            z_values, test_labels_str, gene_names, test_expression,
            comp, testing_method=testing_method,
            shap_thresh=shap_thresh, logfc_thresh=logfc_thresh,
            expression_threshold=expression_threshold,
            p_adj_thresh=p_adj_thresh,
        )

        if len(result) > 0:
            all_results[name] = result
            result.to_csv(
                os.path.join(output_dir, f"{prefix}dxg_{name}.csv"),
                index=False
            )

    print(f"DXG analysis complete. {len(all_results)} comparisons with results.")
    return all_results
