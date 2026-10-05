"""
DEG (Differentially Expressed Genes) analysis — traditional baseline.

Uses Wilcoxon rank-sum test on raw expression values (not attribution scores).
This provides the baseline to compare against DXG results.

Matches the paper's approach of using Seurat's Wilcoxon test with default
filtering parameters.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import ranksums, ttest_ind
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing import List, Dict

from dxg import define_groups, get_all_comparisons


def deg_analysis(
    test_expression: np.ndarray,
    test_labels_str: List[str],
    gene_names: List[str],
    comparison: Dict,
    testing_method: str = "wilcoxon",
    logfc_thresh: float = 0.25,
    p_adj_thresh: float = 0.05,
    min_pct: float = 0.1,
) -> pd.DataFrame:
    """
    Perform traditional DEG analysis for a single comparison.
    Uses Wilcoxon rank-sum test on raw expression values.

    Args:
        test_expression: (N_cells, N_genes) raw expression values
        test_labels_str: label strings
        gene_names: gene names
        comparison: dict with species, cell_type, disease_a, disease_b
        testing_method: "wilcoxon" or "t"
        logfc_thresh: minimum log fold change
        p_adj_thresh: adjusted p-value threshold
        min_pct: minimum fraction of cells expressing gene in either group

    Returns:
        DataFrame with DEG results
    """
    species = comparison['species']
    cell_type = comparison['cell_type']
    disease_a = comparison['disease_a']
    disease_b = comparison['disease_b']

    # Define groups
    mask_a = define_groups(test_labels_str, species, cell_type, disease_a)
    mask_b = define_groups(test_labels_str, species, cell_type, disease_b)

    if mask_a.sum() < 2 or mask_b.sum() < 2:
        return pd.DataFrame()

    expr_a = test_expression[mask_a]  # (n_a, N_genes)
    expr_b = test_expression[mask_b]  # (n_b, N_genes)

    n_genes = test_expression.shape[1]
    results = []

    for g in range(n_genes):
        subset_a = expr_a[:, g]
        subset_b = expr_b[:, g]

        # Minimum percent expressed filter (Seurat default)
        pct_a = (subset_a > 0).mean()
        pct_b = (subset_b > 0).mean()
        if max(pct_a, pct_b) < min_pct:
            continue

        if testing_method == "t":
            stat, p_val = ttest_ind(subset_a, subset_b, equal_var=True)
        else:
            stat, p_val = ranksums(subset_a, subset_b)

        # Log fold change (use log1p of mean ratio)
        mean_a = np.mean(subset_a) + 1e-8
        mean_b = np.mean(subset_b) + 1e-8
        log_fc = np.log2(mean_a / mean_b)

        results.append({
            'Feature': gene_names[g],
            'p_val': p_val,
            'logFC': log_fc,
            'mean_expr_group1': mean_a,
            'mean_expr_group2': mean_b,
            'pct_group1': pct_a,
            'pct_group2': pct_b,
        })

    if len(results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Benjamini-Hochberg correction
    rejected, p_adjusted, _, _ = multipletests(
        results_df['p_val'].values,
        alpha=0.05, method='fdr_bh', is_sorted=False
    )
    results_df['p_adj'] = p_adjusted

    # Filter
    results_df = results_df[results_df['p_adj'] <= p_adj_thresh]
    results_df = results_df[np.abs(results_df['logFC']) >= logfc_thresh]
    results_df = results_df.sort_values('logFC', ascending=False)

    return results_df


def run_all_deg(
    test_expression: np.ndarray,
    test_labels_str: List[str],
    gene_names: List[str],
    output_dir: str,
    testing_method: str = "wilcoxon",
    logfc_thresh: float = 0.25,
    p_adj_thresh: float = 0.05,
    min_pct: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """
    Run DEG analysis for all comparison pairs.

    Returns:
        dict mapping comparison name -> results DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)

    comparisons = get_all_comparisons()
    all_results = {}

    for comp in tqdm(comparisons, desc="DEG comparisons"):
        name = comp['name']
        result = deg_analysis(
            test_expression, test_labels_str, gene_names, comp,
            testing_method=testing_method,
            logfc_thresh=logfc_thresh,
            p_adj_thresh=p_adj_thresh,
            min_pct=min_pct,
        )

        if len(result) > 0:
            all_results[name] = result
            result.to_csv(
                os.path.join(output_dir, f"deg_{name}.csv"),
                index=False
            )

    print(f"DEG analysis complete. {len(all_results)} comparisons with results.")
    return all_results


def compute_deg_score(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GSEA score per gene for DEG results.
    Score = sign(FC) * log10(|FC|/p_bin + 1)
    where p_bin = 0.1 if p < 0.05, else 1.

    This matches the paper's GSEA score formula.
    """
    df = results_df.copy()
    # p_bin: 0.1 if p_adj < 0.05, else 1
    df['p_bin'] = np.where(df['p_adj'] < 0.05, 0.1, 1.0)
    # GSEA score
    df['gsea_score'] = np.sign(df['logFC']) * np.log10(
        np.abs(df['logFC']) / df['p_bin'] + 1
    )
    return df
